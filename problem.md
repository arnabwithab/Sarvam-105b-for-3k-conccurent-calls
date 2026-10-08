### Overview
We have to design an architecture with the following resources

- LLM: Sarvam 105b
- 3000 concurrent calls
- voice agents, multiturn in nature
- 100 b200 GPUs
- 600 RPS
- vLLM servers

#### We have to achieve
- 600ms TTFT

#### Already exists
- TPOT 10-15 ms

#### We have to optimize towards
- High cache hit rate
- Latency

#### We have to only optimize
- LLM serving
- no need to optimize telephony, stt, tts, etc.

#### Submittable
- a PNG with a write up

can use https://github.com/mingrammer/diagrams for drawing