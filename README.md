## MACS: LLM-Enhanced Multi-AUV Collaborative Search Scheme via Multi-agent Reinforcement Learning Approach

*Peijun Dong, Hanjiang Luo, Hang Tao, Wei Shi, Jingjing Wang, Jiehan Zhou*

#### Abstract

Multiple autonomous underwater vehicles (AUVs) integrating multi-agent reinforcement learning (MARL) frame work have made remarkable achievement and widely utilized for underwater search and rescue missions. However, to per form collaborative multi-AUV search efficiently in harsh and communication-constrained marine environments, challenging issues need to be addressed, such as cold start problem and poor collaborative information fusion. To deal with these chal lenges, this paper proposes a MACS scheme which integrates the comprehension and reasoning capabilities of large language models (LLMs) into the MARL framework to solve the cold start problem and facilitate efficient collaborative information fusion. In MACS, to alleviate the cold-start problem of MARL caused by the lack of prior knowledge, we design a LEMACS algorithm, in which we leverage LLMs to infer the initial Target Probability Map (TPM) from search tasks and underwater terrain informa tion to accelerate the search process. Furthermore, to address low efficient data exchange and fusion issue under unstable channel, we propose a LLM-enhanced link selection algorithm LESCLwhich integrates TPM information and AUV link metrics to optimize the link selection procedure to enhance multi-AUV cooperative search information fusion. To validate the effective ness of the proposed algorithms, we conduct extensive numerical simulations using open-source regional underwater terrain data, such as coral reef map dataset of Arizona State University (ASU) and the terrain data of the Dongsha Islands, and the simulation results indicate that MACS achieves a search success rate of up to 95% in emergency multi-AUV cooperative search missions. 

#### Python programming

1. Install dependencies

`pip install -r requirements.txt`

2. Obtain the API KEY of LLM

`client = OpenAI(
        api_key="OPENAI_API_KEY",
        base_url="OPENAI_BASE_URL")`

3. Training

`python train_LEMACS.py`

`python train_LESCL.py`

