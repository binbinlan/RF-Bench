# ![Image text](https://github.com/binbinlan/RF-Bench/blob/main/pics/fig4.png) RF-Bench
***
A deep learning based benchmark for hourly runoff and flood forecasting
***
<div align=center><img src="https://github.com/binbinlan/RF-Bench/blob/main/pics/fig2.png/"></div>

<br/> 

* Flood forecasts with daily resolution are challenged to capture the rapid changes in runoff over short periods. To address this, this paper proposes a benchmark evaluation for runoff and flood forecasting based on deep learning (RF-Bench) and conducts a large-scale fair comparison of various deep learning models at an hourly scale for the first time. The study utilizes data from 516 catchments in the CAMELS datasets and incorporates various models, including Dlinear, LSTM, Transformer and its improved version (Informer, Autoformer, Patch Transformer), and state-space models (Mamba), for benchmarks. Results indicate that the Patch Transformer exhibits optimal predictive capability across multiple lead times, while the traditional LSTM model demonstrates stable performance, and the Mamba model strikes a good balance between performance and stability.
* By analyzing the attention mechanism, the study reveals the attention patterns of Transformer models in hydrological modeling, finding that attention is time-sensitive and that the attention scores for dynamic variables are higher than those for static attributes. Furthermore, experiments further validate the advantages of deep learning models in peak flow prediction and under extreme conditions.
* The construction of RF-Bench provides the hydrological community with an open-source, scalable platform, contributing to the advancement of deep learning in the field of hydrology.
* <u>[(https://colab.research.google.com/)]<u>





