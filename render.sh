#!/bin/bash

pandoc content.md -c hugo-octopress.css -s --mathjax -o index.html
refresh-chrome content.html
