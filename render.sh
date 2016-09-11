#!/bin/bash

pandoc content.md -c github-markdown.css -s --mathjax -o content.html
refresh-chrome content.html
