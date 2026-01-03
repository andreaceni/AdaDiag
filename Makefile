.DEFAULT_GOAL := install

install:
	@echo "Installing..."
	pip install -r requirements.txt
	pip install poetry
	git clone https://github.com/vdecaro/torchdyno
	cd torchdyno; git checkout fix/rnn-of-rnns; poetry install
	@echo "Installation complete."