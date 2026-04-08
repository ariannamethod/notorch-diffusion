"""
notorch_wrapper.py — Python ctypes bindings for notorch (ariannamethod/notorch)

Provides HeVLMModel class for inference using the notorch C library.
Training is done via the C executable (train_hevlm); this wrapper handles
loading trained weights and running generation from Python.
"""

import ctypes
import os
import struct
import subprocess
import sys
import math
import numpy as np

# ── Model config (must match train_hevlm.c) ──

V = 256        # byte-level vocab
E = 128        # embedding dim
H = 4          # attention heads
HD = E // H    # 32 head dim
FFN = 512      # FFN hidden
CTX = 64       # context length
N_LAYERS = 4   # transformer layers


def _lib_dir():
    return os.path.dirname(os.path.abspath(__file__))


def build_notorch():
    """Build the notorch shared library and training executable."""
    lib_dir = _lib_dir()
    print(f"Building notorch in {lib_dir}...")
    result = subprocess.run(
        ["make", "all"],
        cwd=lib_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Build stdout: {result.stdout}")
        print(f"Build stderr: {result.stderr}")
        raise RuntimeError("Failed to build notorch")
    print("notorch built successfully.")


class NotorchLib:
    """Thin ctypes wrapper around libnotorch.so."""

    def __init__(self):
        lib_dir = _lib_dir()
        lib_path = os.path.join(lib_dir, "libnotorch.so")
        if not os.path.exists(lib_path):
            build_notorch()
        self.lib = ctypes.CDLL(lib_path)
        self._setup_types()

    def _setup_types(self):
        L = self.lib

        # nt_tensor* nt_tensor_new(int len)
        L.nt_tensor_new.argtypes = [ctypes.c_int]
        L.nt_tensor_new.restype = ctypes.c_void_p

        # nt_tensor* nt_tensor_new2d(int rows, int cols)
        L.nt_tensor_new2d.argtypes = [ctypes.c_int, ctypes.c_int]
        L.nt_tensor_new2d.restype = ctypes.c_void_p

        # void nt_tensor_free(nt_tensor* t)
        L.nt_tensor_free.argtypes = [ctypes.c_void_p]
        L.nt_tensor_free.restype = None

        # void nt_tensor_fill(nt_tensor* t, float val)
        L.nt_tensor_fill.argtypes = [ctypes.c_void_p, ctypes.c_float]
        L.nt_tensor_fill.restype = None

        # void nt_seed(uint64_t seed)
        L.nt_seed.argtypes = [ctypes.c_uint64]
        L.nt_seed.restype = None

        # void nt_train_mode(int training)
        L.nt_train_mode.argtypes = [ctypes.c_int]
        L.nt_train_mode.restype = None

        # nt_tensor** nt_load(const char* path, int* n_params)
        L.nt_load.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
        L.nt_load.restype = ctypes.c_void_p

        # int nt_save(const char* path, nt_tensor** params, int n_params)
        L.nt_save.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int]
        L.nt_save.restype = ctypes.c_int

        # Tape ops
        L.nt_tape_start.argtypes = []
        L.nt_tape_start.restype = None
        L.nt_tape_clear.argtypes = []
        L.nt_tape_clear.restype = None
        L.nt_tape_get.argtypes = []
        L.nt_tape_get.restype = ctypes.c_void_p

        L.nt_tape_param.argtypes = [ctypes.c_void_p]
        L.nt_tape_param.restype = ctypes.c_int

        L.nt_tape_no_decay.argtypes = [ctypes.c_int]
        L.nt_tape_no_decay.restype = None

        L.nt_tape_record.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_float,
        ]
        L.nt_tape_record.restype = ctypes.c_int

        L.nt_seq_embedding.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
        ]
        L.nt_seq_embedding.restype = ctypes.c_int

        L.nt_seq_linear.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        L.nt_seq_linear.restype = ctypes.c_int

        L.nt_seq_rmsnorm.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        L.nt_seq_rmsnorm.restype = ctypes.c_int

        L.nt_mh_causal_attention.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
        ]
        L.nt_mh_causal_attention.restype = ctypes.c_int

        L.nt_silu.argtypes = [ctypes.c_int]
        L.nt_silu.restype = ctypes.c_int

        L.nt_mul.argtypes = [ctypes.c_int, ctypes.c_int]
        L.nt_mul.restype = ctypes.c_int

        L.nt_add.argtypes = [ctypes.c_int, ctypes.c_int]
        L.nt_add.restype = ctypes.c_int

        L.nt_seq_cross_entropy.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        L.nt_seq_cross_entropy.restype = ctypes.c_int

        L.nt_tape_backward.argtypes = [ctypes.c_int]
        L.nt_tape_backward.restype = None

        L.nt_tape_clip_grads.argtypes = [ctypes.c_float]
        L.nt_tape_clip_grads.restype = ctypes.c_float

        L.nt_tape_chuck_step.argtypes = [ctypes.c_float, ctypes.c_float]
        L.nt_tape_chuck_step.restype = None


# ── nt_tensor struct layout (for reading data pointer) ──
# Must match notorch.h: float* data, int ndim, int shape[8], int stride[8], int len, int refcount
NT_MAX_DIMS = 8


class NtTensorStruct(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_float)),
        ("ndim", ctypes.c_int),
        ("shape", ctypes.c_int * NT_MAX_DIMS),
        ("stride", ctypes.c_int * NT_MAX_DIMS),
        ("len", ctypes.c_int),
        ("refcount", ctypes.c_int),
    ]


def _tensor_to_numpy(tensor_ptr):
    """Read an nt_tensor* into a numpy array."""
    t = ctypes.cast(tensor_ptr, ctypes.POINTER(NtTensorStruct)).contents
    arr = np.ctypeslib.as_array(t.data, shape=(t.len,)).copy()
    return arr


# ── Weight loading (binary format matching nt_save/nt_load) ──

WEIGHT_MAGIC = 0x4E545748  # "NTWH" — notorch weights header


def load_weights_numpy(path):
    """Load notorch binary weights into list of numpy arrays."""
    with open(path, "rb") as f:
        magic, n = struct.unpack("II", f.read(8))
        if magic != WEIGHT_MAGIC:
            raise ValueError(f"Bad magic: 0x{magic:08X} (expected 0x{WEIGHT_MAGIC:08X})")
        tensors = []
        for _ in range(n):
            (ndim,) = struct.unpack("i", f.read(4))
            shape = struct.unpack(f"{ndim}i", f.read(4 * ndim))
            total = 1
            for s in shape:
                total *= s
            data = np.frombuffer(f.read(4 * total), dtype=np.float32).copy()
            data = data.reshape(shape)
            tensors.append(data)
    return tensors


class HeVLMModel:
    """HeVLM model for inference — loads weights and generates text.

    Uses pure numpy for the forward pass (no notorch C library needed for inference).
    This makes inference portable and dependency-free.
    """

    def __init__(self, weight_path=None):
        if weight_path is None:
            weight_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "weights", "hevlm.bin",
            )
        self.weights = load_weights_numpy(weight_path)
        self._parse_weights()
        n = sum(w.size for w in self.weights)
        print(f"HeVLM loaded: {n:,} parameters from {weight_path}")

    def _parse_weights(self):
        """Map flat weight list to named parameters."""
        w = self.weights
        idx = 0
        self.wte = w[idx]; idx += 1   # [V, E]
        self.wpe = w[idx]; idx += 1   # [CTX, E]
        self.layers = []
        for _ in range(N_LAYERS):
            layer = {
                "rms1": w[idx], "wq": w[idx+1], "wk": w[idx+2],
                "wv": w[idx+3], "wo": w[idx+4], "rms2": w[idx+5],
                "w_gate": w[idx+6], "w_up": w[idx+7], "w_down": w[idx+8],
            }
            idx += 9
            self.layers.append(layer)
        self.rms_f = w[idx]; idx += 1  # [E]
        self.head = w[idx]; idx += 1   # [V, E]

    @staticmethod
    def _rmsnorm(x, gamma):
        rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6)
        return (x / rms) * gamma

    @staticmethod
    def _silu(x):
        return x * (1.0 / (1.0 + np.exp(-np.clip(x, -20, 20))))

    @staticmethod
    def _softmax(x):
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / np.sum(e, axis=-1, keepdims=True)

    def _attention(self, q, k, v, T):
        """Multi-head causal attention."""
        # q,k,v: [T, E] → reshape to [T, H, HD]
        q = q.reshape(T, H, HD)
        k = k.reshape(T, H, HD)
        v = v.reshape(T, H, HD)

        out = np.zeros((T, H, HD), dtype=np.float32)
        for h in range(H):
            # [T, HD] @ [HD, T] = [T, T]
            scores = q[:, h, :] @ k[:, h, :].T / math.sqrt(HD)
            # Causal mask
            mask = np.triu(np.full((T, T), -1e9, dtype=np.float32), k=1)
            scores += mask
            attn = self._softmax(scores)
            out[:, h, :] = attn @ v[:, h, :]
        return out.reshape(T, E)

    def forward(self, token_ids):
        """Forward pass. token_ids: list of int byte values. Returns logits [T, V]."""
        T = len(token_ids)
        assert T <= CTX

        # Embedding
        h = self.wte[token_ids] + self.wpe[:T]

        # Transformer blocks
        for layer in self.layers:
            # Attention
            xn = self._rmsnorm(h, layer["rms1"])
            q = xn @ layer["wq"].T   # [T, E] @ [E, E]
            k = xn @ layer["wk"].T
            v = xn @ layer["wv"].T
            attn = self._attention(q, k, v, T)
            proj = attn @ layer["wo"].T
            h = h + proj

            # FFN
            xn = self._rmsnorm(h, layer["rms2"])
            gate = xn @ layer["w_gate"].T  # [T, FFN]
            up = xn @ layer["w_up"].T
            gate = self._silu(gate)
            ffn_out = (gate * up) @ layer["w_down"].T
            h = h + ffn_out

        # Final norm + head
        h = self._rmsnorm(h, self.rms_f)
        logits = h @ self.head.T  # [T, V]
        return logits

    def generate(self, seed_bytes, max_tokens=50, temperature=0.8, top_k=40):
        """Generate text from seed bytes."""
        ctx = list(seed_bytes[:CTX])
        generated = []

        for _ in range(max_tokens):
            if len(ctx) >= CTX:
                break
            logits = self.forward(ctx)
            last_logits = logits[-1] / temperature

            # Top-k sampling
            if top_k > 0:
                top_indices = np.argsort(last_logits)[-top_k:]
                mask = np.full(V, -1e9, dtype=np.float32)
                mask[top_indices] = last_logits[top_indices]
                last_logits = mask

            probs = self._softmax(last_logits)
            next_byte = np.random.choice(V, p=probs)
            ctx.append(int(next_byte))
            generated.append(int(next_byte))

        return bytes(generated)
