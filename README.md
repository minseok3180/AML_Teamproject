# AML_Teamproject
2025-spring-DATA303

# Dataset
1. StackedMNIST
https://paperswithcode.com/dataset/stacked-mnist

2. FFHQ256
https://www.kaggle.com/datasets/denislukovnikov/ffhq256-images-only 


# Requirement

- python 3.7.10
- torch == 1.7.1 + cu110

- click
- requests
- psutil
- scipy
- tqdm
- ninja


# gitconfig
git config user.name "Minseok Kwon"
git config user.email "als3180@gmail.com"

git config user.name "Jiyoung Hwang"
git config user.email "hjyhera@gmail.com"

git config user.name "Taegyu Hwang"
git config user.email "yyggh337@gmail.com"

# 명령어 정리
# 백그라운드에서 시작
nohup torchrun --nproc_per_node=4 main.py > output.log 2>&1 &

# 돌아가는 과정 (원래 터미널에 뜨던 것들) 확인
output.log파일 보기

# 현재 돌아가는 백그라운드 종료
kill -9 <PID>