#!/usr/bin/env python3
"""
AVSE Backend with Metrics
- Consistent framing (b"End"), handshake config from client
- Face detection (grayscale) -> 224x224 face frames
- ONNX Runtime with optimisations (+ optional CUDA)
- Metrics: SNR in/out/Δ (blind), SI-SDR & PESQ* (optional), NN latency,
  one-way latency (client timestamp), throughput (in/out), packet loss
* SI-SDR & PESQ only if a clean reference is provided in the packet as 'ref'
"""

import argparse
import logging
import math
import socket
import pickle
import time
from collections import deque
from multiprocessing import Process, Queue
from typing import Deque, Dict, Any, Tuple, Optional

import numpy as np
import cv2
import librosa
import onnxruntime
# Optional objective metric (only used if you send a clean reference)
try:
    from pesq import pesq as pesq_fn  # pip install pesq
except Exception:
    pesq_fn = None

# ---------------- Config ----------------
FS = 16000
FPS = 25
CHUNK = 640                   # 16kHz / 25fps
RESIZE_HW = (224, 224)
PROC_INTERVAL_S = 2.4         # seconds (interval between NN outputs)

END_FLAG = b"End"
NOSIG_FLAG = b"NoSig"

# ---------------- Logging ----------------
def setup_logging(level=logging.INFO):
    fmt = "%(asctime)s | %(processName)s | %(name)s | %(levelname)s | %(message)s"
    logging.basicConfig(level=level, format=fmt)
setup_logging()

# ---------------- DSP utils ----------------
def divide_magphase(D, power=1):
    mag = np.abs(D)
    mag **= power
    phase = np.exp(1.j * np.angle(D))
    return mag, phase

def merge_magphase(magnitude, phase):
    return magnitude * phase

def estimate_snr_db_blind(sig: np.ndarray, fs: int = FS) -> float:
    """Blind SNR using median noise PSD proxy in STFT domain."""
    eps = 1e-12
    S = librosa.stft(sig.astype(np.float32), n_fft=256, hop_length=128, window="hann")
    mag, _ = divide_magphase(S, power=1)
    pxx = mag**2
    npsd = np.maximum(np.median(pxx, axis=1, keepdims=True), eps)
    snr_lin = np.sum(pxx) / np.sum(npsd)
    return float(10.0 * np.log10(max(snr_lin, eps)))

def si_sdr_db(ref: np.ndarray, est: np.ndarray) -> Optional[float]:
    if ref is None or est is None or len(ref) == 0 or len(est) == 0:
        return None
    L = min(len(ref), len(est))
    s = ref[:L].astype(np.float64)
    x = est[:L].astype(np.float64)
    eps = 1e-12
    alpha = np.dot(x, s) / (np.dot(s, s) + eps)
    e_target = alpha * s
    e_noise = x - e_target
    num = np.sum(e_target**2)
    den = np.sum(e_noise**2) + eps
    return float(10.0 * np.log10(max(num / den, eps)))

def pesq_wb(ref: np.ndarray, deg: np.ndarray, fs: int = FS) -> Optional[float]:
    if pesq_fn is None:
        return None
    try:
        L = min(len(ref), len(deg))
        return float(pesq_fn(fs, ref[:L], deg[:L], 'wb'))
    except Exception:
        return None

def mmse_audio_denoise(noisy_audio: np.ndarray, fs: int = FS) -> np.ndarray:
    """Lightweight MMSE estimator (no SciPy dependency)."""
    NFFT, hop = 256, 128
    eps = 1e-12
    S = librosa.stft(noisy_audio.astype(np.float32), n_fft=NFFT, hop_length=hop, window="hamming")
    mag, ph = divide_magphase(S)
    pxx = mag**2
    npsd = np.maximum(np.median(pxx, axis=1, keepdims=True), eps)
    aPost = np.clip(pxx / (npsd + eps), 1.0, 100.0)
    apri = aPost / (1.0 + aPost)
    V = apri * aPost / (1.0 + apri)
    gain = apri.copy()
    idx = V < 1.0
    if np.any(idx):
        Vi = V[idx]
        aPi = aPost[idx]
        def I0(x): y = x/2.0; return 1.0 + y*y + (y**4)/4.0 + (y**6)/36.0 + (y**8)/576.0
        def I1(x): y = x/2.0; return y*(1.0 + (y**2)/2.0 + (y**4)/12.0 + (y**6)/144.0)
        gain[idx] = (math.gamma(1.5)*np.sqrt(Vi))/(aPi+eps)*np.exp(-Vi/2.0)*((1.0+Vi)*I0(Vi/2.0)+Vi*I1(Vi/2.0))
    out = librosa.istft(merge_magphase(gain*mag, ph), hop_length=hop, window="hamming")
    if len(out) < len(noisy_audio):
        out = np.pad(out, (0, len(noisy_audio)-len(out)))
    else:
        out = out[:len(noisy_audio)]
    return np.clip(out.astype(np.float32), -1.0, 1.0)

# ---------------- Data formatting ----------------
def make_onnx_inputs(audio_bytes: bytes, face_stack: Deque[np.ndarray]) -> Dict[str, np.ndarray]:
    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0  # [T]
    frames = np.array(face_stack, dtype=np.float32)                                  # [F,H,W]
    return {"noisy_audio": audio[np.newaxis, :], "video_frames": frames[np.newaxis, np.newaxis, :, :, :]}

# ---------------- Processes ----------------
def buffer_processing(qin: Queue, qout: Queue, rxnum: Queue, buffer: Queue, proc_info: Queue,
                      face_detection: bool, resize_hw: Tuple[int,int]):
    log = logging.getLogger("buffer")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while proc_info.empty(): time.sleep(0.01)
    config = proc_info.get(); proc_info.put(config)

    compress_flag: bool = bool(config.get('compress_flag', False))
    proc_duration: float = float(config.get('proc_duration', 3.0))
    proc_frame = int(FPS * proc_duration)
    interval_frame = int(FPS * PROC_INTERVAL_S)
    non_nnproc_frame = proc_frame - interval_frame

    audio_buf = b""
    faces_buf: Deque[np.ndarray] = deque(maxlen=proc_frame)

    expected_seq: Optional[int] = None
    lost = 0; received = 0
    bytes_in_window = 0; t0 = time.time()

    last_oneway_ms = None

    log.info(f"Buffer started | compress={compress_flag} | duration={proc_duration}s")

    while True:
        if qin.empty():
            time.sleep(0.0005)
            # periodic throughput report
            if time.time() - t0 >= 1.0:
                mbps = (bytes_in_window*8)/1e6
                log.info(f"IN throughput: {mbps:.3f} Mbps | rx={received} pkts | loss={lost}")
                bytes_in_window = 0; received = 0; lost = 0; t0 = time.time()
            continue

        num_rx_time = rxnum.get()
        item = qin.get()
        bytes_in_window += item.get('_bytes', 0)
        received += 1

        # packet loss from seq gaps
        seq = item.get('seq', None)
        if seq is not None:
            if expected_seq is None:
                expected_seq = seq + 1
            else:
                if seq != expected_seq:
                    if seq > expected_seq:
                        lost += (seq - expected_seq)
                    expected_seq = seq + 1
                else:
                    expected_seq += 1

        # one-way latency (client timestamp)
        ts = item.get('ts', None)
        if ts is not None:
            last_oneway_ms = (time.time() - ts) * 1000.0

        # Face detection (always grayscale)
        if compress_flag:
            gray = cv2.imdecode(np.frombuffer(item['video'], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        else:
            src = item['video']
            gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

        if face_detection:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(faces) > 0:
                x, y, w, h = faces[0]
                roi = gray[y:y+h, x:x+w]
                face = cv2.resize(roi, (resize_hw[1], resize_hw[0]))
            else:
                face = cv2.resize(gray, (resize_hw[1], resize_hw[0]))
        else:
            face = cv2.resize(gray, (resize_hw[1], resize_hw[0]))

        faces_buf.append(face)

        # Rolling audio
        if num_rx_time < proc_frame:
            audio_buf += item['audio']
        else:
            L = len(item['audio'])
            audio_buf = audio_buf[L:] + item['audio']

        # Passthrough until first full NN window
        if num_rx_time < non_nnproc_frame:
            qout.put({'audio': item['audio'], 'meta': {'type': 'passthrough'}})

        # Trigger a window at interval
        if num_rx_time > proc_frame-1 and (num_rx_time - proc_frame) % interval_frame == 0:
            buffer.put({
                'audio': audio_buf,
                'video': list(faces_buf),
                'seq': seq,
                'oneway_ms': last_oneway_ms
            })

def AVSE_processing(qout: Queue, buffer: Queue, proc_info: Queue,
                    use_cuda: bool, use_mmse: bool):
    log = logging.getLogger("avse")

    available = onnxruntime.get_available_providers()
    providers = (["CUDAExecutionProvider"] if use_cuda and "CUDAExecutionProvider" in available else []) + ["CPUExecutionProvider"]
    so = onnxruntime.SessionOptions()
    so.intra_op_num_threads = 1
    so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_session = onnxruntime.InferenceSession("models/model.onnx", providers=providers, sess_options=so)
    log.info(f"ORT available: {available} | using: {providers}")
    log.info("Neural network loaded, ready for AVSE processing...")

    while proc_info.empty(): time.sleep(0.01)
    config = proc_info.get(); proc_info.put(config)

    proc_duration = float(config.get('proc_duration', 3.0))
    proc_frame = int(FPS * proc_duration)
    interval_frame = int(FPS * PROC_INTERVAL_S)
    non_nnproc_frame = proc_frame - interval_frame

    while True:
        if buffer.empty():
            time.sleep(0.0005)
            continue

        pkt = buffer.get()
        seq = pkt.get('seq', None)
        oneway_ms = pkt.get('oneway_ms', None)

        # Prepare inputs
        datain = make_onnx_inputs(pkt['audio'], deque(pkt['video']))

        # NN inference timing
        t0 = time.perf_counter()
        enhanced = ort_session.run(None, datain)[0][0].astype(np.float32)
        if use_mmse:
            enhanced = mmse_audio_denoise(enhanced, fs=FS)
        t1 = time.perf_counter()
        nn_ms = (t1 - t0) * 1000.0

        # Metrics: SNR in/out/Δ (blind)
        noisy = np.frombuffer(pkt['audio'], dtype=np.int16).astype(np.float32) / 32768.0
        snr_in = estimate_snr_db_blind(noisy, FS)
        snr_out = estimate_snr_db_blind(enhanced, FS)
        delta_snr = snr_out - snr_in

        # Optional objective metrics if 'ref' is provided (float32 [-1,1])
        sisdr = None
        pesq_score = None
        # ref = pkt.get('ref', None)
        # if ref is not None:
        #     sisdr = si_sdr_db(ref, enhanced)
        #     pesq_score = pesq_wb(ref, enhanced, fs=FS)

        out_i16 = (np.clip(enhanced, -1.0, 1.0) * 32768.0).astype(np.int16)

        meta = {
            'type': 'enhanced',
            'seq': seq,
            'nn_ms': nn_ms,
            'snr_in_db': snr_in,
            'snr_out_db': snr_out,
            'delta_snr_db': delta_snr,
            'si_sdr_db': sisdr,
            'pesq_wb': pesq_score,
            'oneway_ms': oneway_ms
        }

        # send the interval portion (aligned with passthrough)
        for i in range(interval_frame):
            start = (i + non_nnproc_frame) * CHUNK
            stop = start + CHUNK
            payload = out_i16[start:stop].tobytes()
            qout.put({'audio': payload, 'meta': meta if i == 0 else None})

        log.info(
            "NN={:.1f} ms | SNRin={:.2f} dB | SNRout={:.2f} dB | ΔSNR={:.2f} dB{}".format(
                nn_ms, snr_in, snr_out, delta_snr,
                f" | 1-way={oneway_ms:.1f} ms" if oneway_ms is not None else ""
            )
        )

def tcp_transceiver(qin: Queue, qout: Queue, rxnum: Queue, proc_info: Queue,
                    host: str, port: int):
    log = logging.getLogger("tcp")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.listen(1)
    log.info(f"Server listening on {host}:{port} ...")

    c, addr = s.accept()
    log.info(f"Connection from {addr}")

    # Handshake
    rx = b""
    while True:
        p = c.recv(4096)
        if not p: continue
        rx += p
        if rx.endswith(END_FLAG):
            cfg = pickle.loads(rx[:-len(END_FLAG)])
            proc_info.put(cfg)
            log.info(f"Config: {cfg}")
            RX_SIZE = int(cfg.get('data_size', 65536))
            break

    # Throughput counters (per second)
    in_bytes = 0; out_bytes = 0
    rx_pkts = 0; tx_pkts = 0
    t0 = time.time()

    while True:
        # ---- Receive one framed message ----
        rxdata = b""
        while True:
            pkt = c.recv(RX_SIZE)
            if not pkt: break
            rxdata += pkt
            if rxdata.endswith(END_FLAG):
                body = rxdata[:-len(END_FLAG)]
                if body == NOSIG_FLAG:
                    break
                try:
                    item = pickle.loads(body)
                except Exception as e:
                    log.error(f"Unpickle rx error: {e}")
                    item = None
                if item is not None:
                    # annotate size for throughput
                    item['_bytes'] = len(body)
                    in_bytes += len(body); rx_pkts += 1
                    rxnum.put(rx_pkts-1)
                    qin.put(item)
                break

        # ---- Transmit enhanced audio (or nosig) ----
        if not qout.empty():
            out_item = qout.get()
            payload = pickle.dumps(out_item)
            c.sendall(payload + END_FLAG)
            out_bytes += len(payload); tx_pkts += 1
        else:
            c.sendall(NOSIG_FLAG + END_FLAG)
            out_bytes += len(NOSIG_FLAG)

        # ---- Report throughput once per second ----
        if time.time() - t0 >= 1.0:
            in_mbps = (in_bytes*8)/1e6
            out_mbps = (out_bytes*8)/1e6
            log.info(f"NET in={in_mbps:.3f} Mbps ({rx_pkts} pkts) | out={out_mbps:.3f} Mbps ({tx_pkts} pkts)")
            in_bytes = out_bytes = 0; rx_pkts = tx_pkts = 0; t0 = time.time()

def main():
    parser = argparse.ArgumentParser(description="AVSE Backend with Metrics")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--cuda", action="store_true", help="Try CUDA EP if available")
    parser.add_argument("--mmse", action="store_true", help="Enable MMSE post-denoise")
    parser.add_argument("--no-fd", dest="face_detection", action="store_false", help="Disable face detection")
    args = parser.parse_args()

    qin = Queue(maxsize=128)
    qout = Queue(maxsize=256)
    buffer = Queue(maxsize=32)
    rxnum = Queue(maxsize=128)
    proc_info = Queue(maxsize=4)

    p_buffer = Process(target=buffer_processing, args=(qin, qout, rxnum, buffer, proc_info, args.face_detection, RESIZE_HW), name="buffer")
    p_avse   = Process(target=AVSE_processing, args=(qout, buffer, proc_info, args.cuda, args.mmse), name="avse")
    p_tcp    = Process(target=tcp_transceiver, args=(qin, qout, rxnum, proc_info, args.host, args.port), name="tcp")

    p_buffer.start(); p_avse.start(); p_tcp.start()
    p_buffer.join();  p_avse.join();  p_tcp.join()

if __name__ == "__main__":
    main()

