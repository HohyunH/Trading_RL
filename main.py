import os
import sys
import logging
import argparse
import json

from utils import *
from learner import *
from data_manage import *

def main():

    os.environ['KERAS_BACKEND'] = 'tensorflow'

    # 출력 경로 설정
    output_path = ('./output/{}_{}_{}'.format(output_name, args.rl_method, args.net))
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    # 파라미터 기록
    with open(os.path.join(output_path, 'params.json'), 'w') as f:
        f.write(json.dumps(vars(args)))
    # 로그 기록 설정
    file_handler = logging.FileHandler(filename=os.path.join(
        output_path, "{}.log".format(output_name)), encoding='utf-8')
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setLevel(logging.DEBUG)
    stream_handler.setLevel(logging.INFO)
    logging.basicConfig(format="%(message)s",
                        handlers=[file_handler, stream_handler], level=logging.DEBUG)

    # 모델 경로 준비
    value_network_path = ''
    policy_network_path = ''
    if value_network_name is not None:
        value_network_path = ('./models/{}.h5'.format(value_network_name))
    else:
        value_network_path = os.path.join(output_path,
                                          '{}_{}_value_{}.h5'.format(args.rl_method, args.net, output_name))
    if policy_network_name is not None:
        policy_network_path = ('./models/{}.h5'.format(policy_network_name))
    else:
        policy_network_path = os.path.join(
            output_path, '{}_{}_policy_{}.h5'.format(
                args.rl_method, args.net, output_name))

    common_params = {}
    list_stock_code = []
    list_chart_data = []
    list_training_data = []
    list_min_trading_unit = []
    list_max_trading_unit = []

    for stock_code in forcodelist:
        # 차트 데이터, 학습 데이터 준비
        chart_data, training_data = load_data(
            ('./data/{}/{}.csv'.format(args.ver, stock_code)), args.start_date,
            args.end_date, ver=args.ver)

        # 최소/최대 투자 단위 설정
        min_trading_unit = max(int(100000 / chart_data.iloc[-1]['close']), 1)
        max_trading_unit = max(int(1000000 / chart_data.iloc[-1]['close']), 1)

        # 공통 파라미터 설정
        common_params = {'rl_method': args.rl_method,
                         'delayed_reward_threshold': args.delayed_reward_threshold,
                         'net': args.net, 'num_steps': args.num_steps, 'lr': args.lr,
                         'output_path': output_path, 'reuse_models': args.reuse_models}

        # 강화학습 시작
        learner = None
        if args.rl_method != 'a3c':
            common_params.update({'stock_code': stock_code,
                                  'chart_data': chart_data,
                                  'training_data': training_data,
                                  'min_trading_unit': min_trading_unit,
                                  'max_trading_unit': max_trading_unit})
            if args.rl_method == 'dqn':
                learner = DQNLearner(**{**common_params,
                                        'value_network_path': value_network_path})
            elif args.rl_method == 'pg':
                learner = PolicyGradientLearner(**{**common_params,
                                                   'policy_network_path': policy_network_path})
            elif args.rl_method == 'ac':
                learner = ActorCriticLearner(**{**common_params,
                                                'value_network_path': value_network_path,
                                                'policy_network_path': policy_network_path})
            elif args.rl_method == 'a2c':
                learner = A2CLearner(**{**common_params,
                                        'value_network_path': value_network_path,
                                        'policy_network_path': policy_network_path})
            if learner is not None:
                learner.run(balance=args.balance,
                            num_epoches=args.epochs,
                            discount_factor=args.discount_factor,
                            start_epsilon=args.start_epsilon,
                            learning=args.learning)
                learner.save_models()
        else:
            list_stock_code.append(stock_code)
            list_chart_data.append(chart_data)
            list_training_data.append(training_data)
            list_min_trading_unit.append(min_trading_unit)
            list_max_trading_unit.append(max_trading_unit)

    if args.rl_method == 'a3c':
        learner = A3CLearner(**{
            **common_params,
            'list_stock_code': list_stock_code,
            'list_chart_data': list_chart_data,
            'list_training_data': list_training_data,
            'list_min_trading_unit': list_min_trading_unit,
            'list_max_trading_unit': list_max_trading_unit,
            'value_network_path': value_network_path,
            'policy_network_path': policy_network_path})

        learner.run(balance=args.balance, num_epoches=args.epochs,
                    discount_factor=args.discount_factor,
                    start_epsilon=args.start_epsilon,
                    learning=args.learning)
        learner.save_models()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--ver', type=str, default="v2", help='The version of rl traders')
    parser.add_argument('--rl_method', type=str, default="a2c", help='The method of RL, DQN, PG, AC, A2C, A3C')
    parser.add_argument('--net', type=str, default='dnn', help='Choose the type of neural network to use in the value neural network and policy neural network.')
    parser.add_argument('--num_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default= 0.01, help='Input the learning rate')
    parser.add_argument('--discount_factor', type=float, default=0.9)
    parser.add_argument('--start_epsilon', type=int, default=0, choices=[0,1], help='Determine the starting exploration rate. As the epoch is performed, the exploration rate decreases.')
    parser.add_argument('--balance', type=int, default=1000000, help='Initial capital for stock investment simulation.')
    parser.add_argument('--epochs', type=int, default=100, help='test 1')
    parser.add_argument('--delayed_reward_threshold', type=float, default=0.04)
    parser.add_argument('--reuse_models', type=bool, default=False, help='train = false, test = true')
    parser.add_argument('--learning', type=bool, default=True, help='train = true, test=false')
    parser.add_argument('--start_date', type=str, default='20170901', help='test=20190902')
    parser.add_argument('--end_date', type=str, default='20190831', help='test=20200831')
    parser.add_argument('--code', type=str, default='kos20', choices=['kos20','kos20_10','bio20','uncontect_20'], help='Enter the transaction item.')

    args = parser.parse_args()

    output_name = get_time_str()
    policy_network_name = args.rl_method
    value_network_name = args.net

    with open('./test.json', 'r', encoding='utf-8-sig') as f:
        json_data = json.load(f)

    forcodelist = []
    for i in json_data[args.code].keys():
      forcodelist.append(i)

    main()