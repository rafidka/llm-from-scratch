CHAPTERS = preface \
          chapter1_tokenization \
          chapter2_attention \
          chapter3_gpt_architecture \
          chapter4_pretraining \
          chapter5_pretrained_weights \
          chapter6_finetuning \
          chapter7_modern_architecture

CHAPTER_PDFS = $(foreach ch,$(CHAPTERS),chapters/pdf/$(ch).pdf)

.PHONY: all chapters clean

all: llm_review_guide.pdf

llm_review_guide.pdf: llm_review_guide.tex preamble.tex $(wildcard chapters/*.tex)
	pdflatex -interaction=nonstopmode llm_review_guide.tex
	pdflatex -interaction=nonstopmode llm_review_guide.tex

define build_chapter
chapters/pdf/$(1).pdf: chapters/$(1).tex preamble.tex
	@mkdir -p chapters/pdf
	@echo "\\\documentclass[11pt,a4paper]{book}" > chapters/pdf/$(1).tex
	@echo "\\\input{../../preamble}" >> chapters/pdf/$(1).tex
	@echo "\\\begin{document}" >> chapters/pdf/$(1).tex
	@echo "\\\input{../$(1)}" >> chapters/pdf/$(1).tex
	@echo "\\\end{document}" >> chapters/pdf/$(1).tex
	cd chapters/pdf && pdflatex -interaction=nonstopmode $(1).tex && pdflatex -interaction=nonstopmode $(1).tex
endef

$(foreach ch,$(CHAPTERS),$(eval $(call build_chapter,$(ch))))

chapters: $(CHAPTER_PDFS)

clean:
	rm -f llm_review_guide.pdf llm_review_guide.aux llm_review_guide.log \
	      llm_review_guide.out llm_review_guide.toc llm_review_guide.lof \
	      llm_review_guide.lot llm_review_guide.synctex.gz
	rm -rf chapters/pdf