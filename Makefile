generate: helper/synthesis.py
	@python helper/synthesis.py

train: analysis/pointwise/train.py
	@python analysis/pointwise/train.py

calibrate_synthetic: analysis/pointwise/calibrate_synthetic.py
	@python analysis/pointwise/calibrate_synthetic.py

calibrate_market: analysis/pointwise/calibrate_market.py
	@python analysis/pointwise/calibrate_market.py

connect: tmp.pem
	@ssh -i fyp.pem ubuntu@ec2-18-141-2-190.ap-southeast-1.compute.amazonaws.com

get_python:
	@curl -O https://bootstrap.pypa.io/get-pip.py
	@python get-pip.py

git:
	@git add plotting/*
	@git add data/*
	@git add weights/*
	@git commit -m "RUN"
	@git push origin

run:
	$(MAKE) train
	$(MAKE) calibrate_synthetic
	$(MAKE) calibrate_market
	$(MAKE) git

env:
	@source /opt/tensorflow/bin/activate


.PHONY = train generate calibrate_synthetic calibrate_market connect get_python run git env