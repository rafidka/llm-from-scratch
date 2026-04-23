.PHONY: all clean

all: llm_review_guide.pdf

llm_review_guide.pdf: llm_review_guide.tex
	pdflatex -interaction=nonstopmode llm_review_guide.tex
	pdflatex -interaction=nonstopmode llm_review_guide.tex

clean:
	rm -f llm_review_guide.pdf llm_review_guide.aux llm_review_guide.log \
	      llm_review_guide.out llm_review_guide.toc llm_review_guide.lof \
	      llm_review_guide.lot llm_review_guide.synctex.gz