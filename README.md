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
