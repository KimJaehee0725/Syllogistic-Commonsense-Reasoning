import argparse

def make_parser() :
    parser = argparse.ArgumentParser(description='BART Train for Syllogistic Generation')
    parser.add_argument(
        '-project_name',
        type = str,
        help = 'Project name',
        default = 'BART Fine-Tuning Generation Only')

    parser.add_argument(
        '-seed',
        type=int,
        default=42)

    parser.add_argument(
        '-data_path', 
        type = str, 
        default = "/datasets/",
        help='학습 데이터 위치')

    parser.add_argument(
        '-model_path', 
        type = str,
        default = "/project/Syllogistic-Commonsense-Reasoning/BART/model_log/",
        help = '모델 저장할 위치')

    parser.add_argument(
        '-generation_path',
        type = str,
        default = "/project/Syllogistic-Commonsense-Reasoning/BART/generation_log/",
        help = '생성된 텍스트 저장할 위치')

    parser.add_argument(
        '-num_epochs',
        type = int,
        default = 7)

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
        default = 4)

    parser.add_argument(
        "-learning_rate",
        type = float,
        default = 3e-5)
    
    parser.add_argument(
        "-log_interval",
        type = int,
        default = 500,
        help = "로그를 출력할 이터레이션 간격")

    parser.add_argument(
        "-eval_interval",
        type = int,
        default = 500,
        help = "평가를 수행할 이터레이션 간격")

    parser.add_argument(
        '-eval_batch_size',
        type = int,
        default = 16)

    parser.add_argument(
        '-save_interval',
        type = int,
        default = 5000,
        help = "모델을 저장할 이터레이션 간격")

    parser.add_argument(
        "-kfold",
        type = int,
        default = 5,
        help = "kfold 개수")

    parser.add_argument(
        '-kfold_idx',
        type = int,
        default = 0,
        help = "kfold 인덱스")

    args = parser.parse_args()
    
    return args
