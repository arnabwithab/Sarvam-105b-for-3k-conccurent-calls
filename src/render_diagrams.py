# /// script
# requires-python = ">=3.10"
# dependencies = ["diagrams"]
# ///

"""Sarvam-105B voice-serving architecture diagrams.

Run with:  uv run diagrams.py

Renders:
  architecture.png            - master diagram (all three planes)
  request_plane.png           - request / serving plane only
  control_plane.png           - control plane only
  observability_plane.png     - observability / SLO plane only
"""

from diagrams import Cluster, Diagram, Edge
from diagrams.generic.compute import Rack
from diagrams.generic.device import Mobile
from diagrams.generic.network import Router
from diagrams.generic.storage import Storage
from diagrams.k8s.compute import Deployment, Pod
from diagrams.k8s.network import Ingress
from diagrams.onprem.monitoring import Grafana, Nagios as Alerting, Prometheus

GRAPH = dict(
    bgcolor="white",
    pad="0.6",
    splines="spline",
    nodesep="0.7",
    ranksep="1.0",
    fontname="Helvetica",
    labelloc="t",
    fontsize="16",
)
NODE = dict(fontname="Helvetica", fontsize="11")
EDGE = dict(fontname="Helvetica", fontsize="9")
GRAPH_WIDE = dict(GRAPH, nodesep="1.5", ranksep="1.1")

GREEN = "#2F855A"   # KV events (cache coherence)
ORANGE = "#C05621"  # KV transfer
PURPLE = "#6B46C1"  # tier-2 offload
GREY = "#718096"

OUT_SCOPE = dict(bgcolor="#F2F2F2", style="dashed", pencolor=GREY)
REQ = dict(bgcolor="#EBF3FA", pencolor="#7FA6C9", style="rounded")
PREFILL = dict(bgcolor="#FFF7E6", pencolor="#D69E2E", style="rounded")
DECODE = dict(bgcolor="#EFF9EE", pencolor="#68A063", style="rounded")
TIER2 = dict(bgcolor="#F6EFFA", pencolor="#9F7AEA", style="rounded")
CTRL = dict(bgcolor="#FFF5F5", pencolor="#C53030", style="rounded")
OBS = dict(bgcolor="#F4F4F4", pencolor="#4A5568", style="rounded")




def build_master():
    """Stripped overview. Detail lives in the per-plane diagrams."""
    with Diagram(
        "Sarvam-105B on 100x B200 - serving overview (TTFT <= 600ms)",
        filename="architecture",
        outformat="png",
        show=False,
        direction="TB",
        graph_attr=GRAPH,
        node_attr=NODE,
        edge_attr=EDGE,
    ):
        with Cluster("Voice input (out of scope)", graph_attr=OUT_SCOPE):
            stt = Mobile("STT final transcripts")
        with Cluster("Voice output (out of scope)", graph_attr=OUT_SCOPE):
            tts = Mobile("TTS stream")

        with Cluster("Request plane", graph_attr=REQ):
            gw = Ingress("Gateway (GAIE)")
            fe = Rack("Dynamo Frontend")
            router = Router("KV-aware Router\nsession affinity")
            with Cluster("Prefill pool - 9x B200", graph_attr=PREFILL):
                pw = Rack("vLLM prefill workers\nTP=1 - cold prompts only")
                pkv = Storage("system-prompt KV")
            with Cluster("Decode pool - 90x B200", graph_attr=DECODE):
                dw = Rack("vLLM decode workers\nTP=1 - FP8 KV - APC")
                dkv = Storage("KV domain (L0)")
            with Cluster("Tier-2 KV (L1)", graph_attr=TIER2):
                l1 = Storage("LMCache / KVBM\nCPU + NVMe")

        with Cluster("Control plane (Dynamo)", graph_attr=CTRL):
            planner = Pod("Planner\nTTFT 600ms - ITL 15ms")
            grove = Deployment("Grove\nplacement")
            mig = Pod("Request\nmigration")

        with Cluster("Data / SLO plane", graph_attr=OBS):
            prom = Prometheus("Prometheus")
            graf = Grafana("Grafana")
            alert = Alerting("SLO alerts\np99 TTFT <600ms")

        stt >> Edge(label="transcript") >> gw >> fe >> router
        router >> Edge(label="new call") >> pw
        router >> Edge(label="warm turn") >> dw
        pw >> pkv
        dw >> dkv
        pw >> Edge(color=ORANGE, label="delta KV (~2ms)") >> dw
        dw >> Edge(color=PURPLE, style="dashed", label="spill / restore",
                   forward=True, reverse=True) >> l1
        dw >> Edge(label="first token") >> tts
        dkv >> Edge(color=GREEN, style="dashed", label="KV events") >> router
        planner >> Edge(style="dashed", label="resize pools") >> router
        dw >> Edge(style="dashed", label="metrics") >> prom
        prom >> graf >> alert


def build_request_plane():
    with Diagram(
        "Request plane - Sarvam-105B voice serving (600 RPS, 3000 calls)",
        filename="request_plane",
        outformat="png",
        show=False,
        direction="TB",
        graph_attr=GRAPH,
        node_attr=NODE,
        edge_attr=EDGE,
    ):
        with Cluster("Voice input (out of scope)", graph_attr=OUT_SCOPE):
            stt = Mobile("STT final transcripts")
        gw = Ingress("K8s Gateway API (GAIE)\nauth - rate-limit")
        fe = Rack("Dynamo Frontend\nOpenAI-compatible - streaming")
        router = Router("Dynamo KV-aware Router\nsession affinity + KV overlap")
        with Cluster("Prefill pool - 9x B200", graph_attr=PREFILL):
            pw = Rack("vLLM prefill workers\nTP=1 - FP8 - 8-16k tok chunks")
            pkv = Storage("shared system-prompt KV")
        with Cluster("Decode pool - 90x B200", graph_attr=DECODE):
            dw = Rack("vLLM decode workers\nTP=1 - FP8 KV 18 KB/tok\nAPC - ~30 convos - ~8 active")
            dkv = Storage("APC KV domain (L0)\n~4M tok headroom (~5% used)")
        with Cluster("Tier-2 KV (L1)", graph_attr=TIER2):
            cpu = Storage("CPU-RAM KV pool\nconversation snapshots")
            nvme = Storage("NVMe / object tier\nlong-context tail")
        with Cluster("Voice output (out of scope)", graph_attr=OUT_SCOPE):
            tts = Mobile("TTS stream")

        stt >> Edge(label="transcript") >> gw >> fe >> router
        router >> Edge(label="new call (1-2.5k tok)") >> pw
        router >> Edge(label="warm turn (50-150 tok)") >> dw
        pw >> pkv
        dw >> dkv
        pw >> Edge(color=ORANGE, label="delta KV - ~45MB - ~2ms") >> dw
        dw >> Edge(color=PURPLE, style="dashed", label="spill / restore",
                   forward=True, reverse=True) >> cpu
        cpu >> Edge(color=PURPLE, style="dashed", label="spill") >> nvme
        dw >> Edge(label="first token") >> tts
        dkv >> Edge(color=GREEN, style="dashed", label="KV block events") >> router


def build_control_plane():
    with Diagram(
        "Control plane - fleet orchestration for 100x B200",
        filename="control_plane",
        outformat="png",
        show=False,
        direction="TB",
        graph_attr=GRAPH_WIDE,
        node_attr=NODE,
        edge_attr=EDGE,
    ):
        with Cluster("GPU fleet: 90 + 9 + 1 spare = 100x B200", graph_attr=REQ):
            pw = Rack("Prefill pool - 9x B200\nvLLM TP=1 - FP8")
            dw = Rack("Decode pool - 90x B200\nTP=1 - FP8 KV - APC")
        with Cluster("Control plane (Dynamo)", graph_attr=CTRL):
            planner = Pod("Dynamo Planner\nTTFT 600ms - ITL 15ms")
            grove = Deployment("Grove operator\ntopology-aware placement")
            disc = Pod("K8s-native discovery\nno etcd / NATS")
            mex = Pod("ModelExpress\n7x faster cold start")
            mig = Pod("Request migration\ncanary health checks")
        planner >> Edge(style="dashed", label="resize 90/9 blend") >> pw
        planner >> Edge(style="dashed") >> dw
        grove >> Edge(style="dashed", label="place pods") >> dw
        mex >> Edge(style="dashed", label="stream weights") >> pw
        mig >> Edge(style="dashed", label="migrate requests") >> dw


def build_observability_plane():
    with Diagram(
        "Observability / SLO plane - what we measure and how we validate",
        filename="observability_plane",
        outformat="png",
        show=False,
        direction="TB",
        graph_attr=GRAPH_WIDE,
        node_attr=NODE,
        edge_attr=EDGE,
    ):
        with Cluster("Metric sources", graph_attr=REQ):
            dw = Rack("Decode pool - 90x B200\nvllm:* + KV events")
            pw = Rack("Prefill pool - 9x B200\nvllm:* engine metrics")
            router = Router("KV-aware Router\nrouting metrics")
        with Cluster("Metrics pipeline", graph_attr=OBS):
            prom = Prometheus("Prometheus\nvllm:* - dynamo:*")
            graf = Grafana("Grafana\ncache-hit - TTFT/ITL\nqueue - KV util")
            alert = Alerting("SLO burn alerting\np99 TTFT <600ms - hit >95%")
            planner = Pod("Dynamo Planner\n(see control plane)")
        with Cluster("Validation (pre-prod)", graph_attr=CTRL):
            sim = Pod("AISimulate\nconfig search")
            perf = Pod("AIPerf + bench serve\n600 RPS replay")
            chaos = Pod("Chaos test\nkill worker\nrecovery TTFT")
        dw >> Edge(style="dashed", label="vllm:* + KV events") >> prom
        pw >> Edge(style="dashed") >> prom
        router >> Edge(style="dashed") >> prom
        prom >> graf >> alert
        alert >> Edge(color=GREY, style="dashed", label="autoscale signal") >> planner
        sim >> Edge(style="dashed", label="sized config") >> planner
        perf >> Edge(style="dashed", label="600 RPS replay") >> router


if __name__ == "__main__":
    build_master()
    build_request_plane()
    build_control_plane()
    build_observability_plane()
    print("rendered: architecture.png, request_plane.png, control_plane.png, observability_plane.png")
