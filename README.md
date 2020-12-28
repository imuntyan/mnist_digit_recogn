# Setup

1. Set local python version

```shell script
pyenv local 3.8.2
```

2. Set up local python environment

```shell script
python -m venv ~/venv/digit_recogn
source ~/venv/digit_recogn/bin/activate
```

To deactivate when done:

```shell script
deactivate
```

# Model fitting

1. Install requirements

```shell script
pip install -r fit/requirements.txt
python fit/mnist_fit.py
``` 

# Web Server

https://www.fullstackpython.com/blog/python-3-flask-green-unicorn-ubuntu-1604-xenial-xerus.html


```shell script
gunicorn web:app
```

