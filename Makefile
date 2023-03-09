generate: helper/synthesis.py
	@python helper/synthesis.py

train: analysis/pointwise/train.py
	@python analysis/pointwise/train.py

calibrate_synthetic: analysis/pointwise/calibrate_synthetic.py
	@python analysis/pointwise/calibrate_synthetic.py

calibrate_market: analysis/pointwise/calibrate_market.py
	@python analysis/pointwise/calibrate_market.py

.PHONY = train generate calibrate_synthetic calibrate_market