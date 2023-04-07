rm -rf .venv
pyenv init -

#pyenv install 3.11.1
pyenv local 3.11.1

python -m venv .venv

source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements/test.txt
source .venv/bin/activate