# Trading_RL

## "강화학습 기반 주식 자동매매 모델 전략 제안"

본 저장소는 "강화학습 기반 주식 자동 매매 모델 전략 제안, 대한산업공학회지 제 47권 제 4호, 황호현 김용훈 이영훈" 의 실험 코드입니다.

### How to use

<pre>
<code>

python main.py --ver v2 --rl_method a2c --net dnn --num_step 1 --lr 0.01 --discount_factor 0.9 --start_epsilon 0 --balance 100000 --epochs 100 --delayed_reward_threshold 0.04 --reuse_models False --learning True --starte_date 20170901 --end_date 20190831 --code kos20 

</code>
</pre>

### Hyper Parameter Setting

|hyper_parameter|options|
|------|---|
|Environment|Stock price data|
|State|Point of Stock price data, Close price moving average ratio|
|Agent|Buying/Selling/Hold|
|Reword|Revenue Beyond Critical Points Loss occurrence|
|Reinforcement learning method|Deep Q Networks, Policy Gradient, Advantage actor-critic|
|Neural network|DNN|
|Learning rate|0.01|
|Initial exploration rate|1|
|Initial capital|10,000,000|
|num of epochs|3|
|Discount factor|0.9|
|Delayed reward threshold|0.04|

### Model Framework

![image](https://user-images.githubusercontent.com/46701548/138587053-26ab5e83-a4ce-4fe6-b983-82853486afc0.png)


### Used items for Datasets

|KOSPI|BIO|Untact|
|------|---|---|
|KB금융|SK바이오랜드|다날|
|LG생활건강|네이처셀|SGA솔루션즈|
|LG전자|동국제약|KG모빌리언스|
|LG화학|랩지노믹스|아이씨케이|
|NAVER|레고켐바이오|이니텍|
|POSCO|비씨월드제약|시큐브|
|SK|삼천당제약|카페24|
|SK텔레콤|신풍제약|NHN한국사이버결제|
|SK하이닉스|수젠텍|인포뱅크|
|기아차|CMG제약|인포바인|
|넷마블|안트로젠|케이씨에스|
|삼성SDI|일신바이오|씨아이테크|
|삼성물산|일양약품|한국전자금융|
|삼성바이오로직스|진원생명과학|윈스|
|삼성전자|차바이오텍|에이택티앤|
|셀트리온|코미팜|파인디지털|
|엔씨소프트|텔콘RF제약|푸른기술|
|카카오|파미셀|디지탈옵틱|
|현대모비스|피씨엘|KG이니시스|
|현대자동차|현대바이오|아이크래프트|


### Experiments
#### 실험1 : 강화학습 기법 성능 비교

![image](https://user-images.githubusercontent.com/46701548/138587331-fb694bfe-b4d6-44e0-a6ec-d4a07c6f73fe.png)

수익성 : A2C / 안정성 : DQN

#### 실험2 : 학습기간의 변동성에 따른 모델 안정성 비교

![image](https://user-images.githubusercontent.com/46701548/138587371-2b50da94-fd23-4441-be28-06fa1d6dcded.png)

회색 : 변동성이 높을 때 테스트 한 "바이오 테마주" / 노란색 : 나머지 5가지 경우에서 같은 결과

![image](https://user-images.githubusercontent.com/46701548/138587446-b6eaa393-8f83-4eff-ab09-eb9fd7c031b2.png)

VIX-low 에서 학습한 모델의 손실평균이 높게 나타남(안정성 낮음)

#### 실험3 : 모델의 학습 기간 길이별 성능 비교

![image](https://user-images.githubusercontent.com/46701548/138587733-4e73e4d0-3dcd-457b-b08e-49cc44d6291c.png)

![image](https://user-images.githubusercontent.com/46701548/138587746-0067b69d-3b1b-4dfc-b9f4-5b1c2306f2af.png)



