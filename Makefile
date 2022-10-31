jupyter: port=8888
jupyter:
	docker build . --tag yolov7
	# Start a jupyter server inside the docker environment of the specified experiment
	docker run --rm -it -p $(port):$(port) $(run-xargs) \
		--mount type=bind,source="$(PWD)",target=/mnt \
		--workdir /mnt \
		yolov7:latest \
		/bin/bash -c "pip install jupyterlab; jupyter lab --allow-root --ip 0.0.0.0 --no-browser --port $(port)" \
		|| true