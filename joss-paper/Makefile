# =============================================================================
# @file    Makefile
# @brief   Makefile for generating previews of the paper
# @author  Michael Hucka <mhucka@caltech.edu>
# @license Please see the file named LICENSE in the project directory
# @website https://github.com/casics/dassie
# =============================================================================

# Change the following values to match your configuration.
# .............................................................................

input	:= paper

vol     := 5
issue   := 54
gh_issue  := 2582
year    := 2020
submitted := 20 July $(year)
accepted  := 22 October $(year)

# Main code -- no more customization variables after this point
# .............................................................................

title	:= $(shell grep title: $(input).md | sed 's/title: *//' | tr -d "'")
authors := $(shell sed -n '/authors:/,/affiliations:/p' $(input).md | grep name: | sed 's/- name: *//' | paste -d, -s - | sed 's/,/, /g')
repo	:= $(shell git remote get-url origin | sed 's|git@github.com:|https://github.com/|' | sed 's/\.git//')

$(input).tex: $(input).md $(input).bib Makefile
	/usr/bin/pandoc \
	-V paper_title="$(title)" \
	-V footnote_paper_title="$(title)" \
	-V citation_author="$(authors)" \
	-V repository="$(repo)" \
	-V archive_doi="http://dx.doi.org/10.21105/zenodo.1400822" \
	-V formatted_doi="10.21105/joss.00850" \
	-V paper_url="http://joss.theoj.org/papers/" \
	-V review_issue_url="https://github.com/openjournals/joss-reviews/issues/$(issue)" \
	-V issue="$(issue)" \
	-V volume="$(vol)" \
	-V year="$(year)" \
	-V submitted="$(submitted)" \
	-V published="$(accepted)" \
	-V page="$(issue)" \
	-V graphics="true" \
	-V logo_path="joss-logo.png" \
	-V geometry:margin=1in \
	--verbose \
	-o $(input).tex \
	--pdf-engine=xelatex \
	--filter /usr/bin/pandoc-citeproc $(input).md \
	--from markdown+autolink_bare_uris \
	--template "latex.template"

$(input).pdf: $(input).tex
	xelatex $(input).tex

autorefresh:;
	((ls $(input).md $(input).bib | entr make $(input).tex) &)
