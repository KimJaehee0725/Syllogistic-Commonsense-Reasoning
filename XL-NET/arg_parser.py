import argparse

def make_parser() :
    parser = argparse.ArgumentParser(description='XL-NET Train for Classification')
    parser.add_argument(
        '-seed',
        type=int,
        default=1234,
    )
    
    parser.add_argument(
        '-data_path', 
        type = str, 
        default = "/datasets/",
        help='학습 데이터 위치')

    parser.add_argument(
        '-model_path', 
        type = str,
        default = "./model_log/",
        help = '모델 저장할 위치')

    parser.add_argument(
        '-num_epochs',
        type = int,
        default = 10)

    parser.add_argument(
        '-num_iteration',
        type = int,
        default = 1000,
        help = "학습을 종료할 목표 이터레이션, epoch보다 우선시 됨")

    parser.add_argument(
        '-max_len',
        type = int,
        default = 512,
        help = "모델의 max len을 따르되 필요 시 수정")
    
    parser.add_argument(
        '-batch_size',
        type = int,
        default = 8)

    parser.add_argument(
        "-learning_rate",
        type = float,
        default = 1e-5)
    
    parser.add_argument(
        "-log_interval",
        type = int,
        default = 100,
        help = "로그를 출력할 이터레이션 간격")

    parser.add_argument(
        "-eval_interval",
        type = int,
        default = 2000,
        help = "평가를 수행할 이터레이션 간격")

    parser.add_argument(
        '-eval_batch_size',
        type = int,
        default = 32)

    parser.add_argument(
        '-save_interval',
        type = int,
        default = 1000,
        help = "모델을 저장할 이터레이션 간격")

    args = parser.parse_args()
    
    return args
