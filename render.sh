#!/bin/bash

pandoc content.md -c hugo-octopress.css -s --mathjax -o content.html
refresh-chrome content.html
