IMAGE ?= blast-radius-entry
SAMPLE := $(CURDIR)/../blast-radius-sample/sample
OUTPUT := $(CURDIR)/output.jsonl

RUN_FLAGS := --rm --platform linux/amd64 --network none --cpus 2 \
	--memory 4096m --memory-swap 4096m --pids-limit 256 --read-only \
	--tmpfs /tmp:rw,size=64m,mode=1777 --cap-drop ALL \
	--security-opt no-new-privileges --user 65534:65534 --env HOME=/tmp \
	--volume $(SAMPLE)/commits:/data/commits:ro \
	--volume $(SAMPLE)/tests.json:/data/tests.json:ro

.PHONY: build sample local

build:
	docker build --platform linux/amd64 --tag $(IMAGE) .

sample: build
	docker run $(RUN_FLAGS) $(IMAGE) > $(OUTPUT)
	python3 ../blast-radius-sample/score.py $(SAMPLE) $(SAMPLE)/answers.json $(OUTPUT)

# Score without Docker. Requires the build-time maps and a checkout of eb11aec.
local:
	PYTHONPATH=$(CURDIR) BLAST_DATA=$(SAMPLE) \
		BLAST_ROOT=$${SQLGLOT_ROOT:-/tmp/sqlglot-src} \
		BLAST_MAP=$${BLAST_MAP:-/tmp/entry2-coverage.json} \
		BLAST_OBSERVED=$${BLAST_OBSERVED:-/tmp/entry2-observed.json} \
		python3 selector.py > $(OUTPUT)
	python3 ../blast-radius-sample/score.py $(SAMPLE) $(SAMPLE)/answers.json $(OUTPUT)
