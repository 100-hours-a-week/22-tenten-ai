# 22-tenten-ai
- [AI wiki 바로가기](https://github.com/100-hours-a-week/22-tenten-wiki/wiki/AI-wiki)
- [22-tenten 프로젝트 로드맵(AI repo) 바로가기](https://github.com/orgs/100-hours-a-week/projects/202/views/2?filterQuery=repo%3A100-hours-a-week%2F22-tenten-ai)


<br/>

# For RUN
## python 3.11.12 변경 (pyenv 다운)
1. python, pip 설치
```bash
sudo apt update
sudo apt install python3
sudo apt install python3-pip
```
2. 필요한 패키지 설치
```bash
sudo apt update
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev curl llvm \
  libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```
3. pyenv 설치
```bash
curl https://pyenv.run | bash
```
4. 셸 설정 파일 수정 (bash 기준)
```bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc
```
5. 원하는 버전 설치
```bash
pyenv install 3.11.12
pyenv global 3.11.12
```

## GCP 서버 이사갔을때 main.py 정상 작동 시키기
1. git clone
2. 가상환경 설정
```bash
python3 -m venv .venv
```
3. 가상환경 활성화
```bash
source .venv/bin/activate
```
4. requirements 다운로드
```bash
pip install -r requirements-gcp.txt
```
5. cuda(version : 12.4) 설정
```bash
sudo apt update
sudo apt install nvidia-driver-550
sudo reboot
```
---
## 가상환경 
### 가상환경 생성
```bash
python3 -m venv .venv
```
### 가상환경 활성화
```bash
source .venv/bin/activate
```
---
## requirements
### gcp
```bash
pip install -r requirements-gcp.txt
```
### local
```bash
pip install -r requirements.txt
```
---
## 실행
### gcp
```bash
python main.py --mode gcp-prod
```
```bash
python main.py --mode gcp-dev
```
### api(gemini)
```bash
python main.py --mode api-prod
```
```bash
python main.py --mode api-dev
```
### local (colab 통신)
```bash
python main.py
```
