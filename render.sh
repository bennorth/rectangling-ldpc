#!/bin/bash

pandoc content.md -c hugo-octopress.css -s --mathjax -o index.html
inkscape factor-graph-chi.svg --export-png=factor-graph-chi.png --export-width=480
inkscape factor-graph-one-chk.svg --export-png=factor-graph-one-chk.png --export-width=420

refresh-chrome content.html
