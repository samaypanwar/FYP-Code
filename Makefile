generate: helper/synthesis.py
	@python helper/synthesis.py

train: analysis/pointwise/train.py
	@python analysis/pointwise/train.py

calibrate_synthetic: analysis/pointwise/calibrate_synthetic.py
	@python analysis/pointwise/calibrate_synthetic.py

calibrate_market: analysis/pointwise/calibrate_market.py
	@python analysis/pointwise/calibrate_market.py

connect: fyp.pem
	@ssh -i fyp.pem ec2-user@ec2-18-142-207-9.ap-southeast-1.compute.amazonaws.com

get_python:
	@curl -O https://bootstrap.pypa.io/get-pip.py
	@python get-pip.py

connect2: fyp2.pem
	@ssh -i fyp2.pem ec2-user@ec2-18-141-2-190.ap-southeast-1.compute.amazonaws.com

git:
	@git add .
	@git commit -m "RUN"
	@git push origin

run:
	$(MAKE) train
	$(MAKE) calibrate_synthetic
	$(MAKE) calibrate_market
	$(MAKE) git

.PHONY = train generate calibrate_synthetic calibrate_market connect get_python run git