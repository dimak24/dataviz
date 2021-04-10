### Requirements
- [pygraphml](http://graphml.graphdrawing.org/) --- a library for parsing graphs in the GraphML format
- [graphviz](https://graphviz.org/) --- graph vizualization software (in this project is used only for drawing graphs by the found coordinates)
- [pygraphviz](https://pygraphviz.github.io/) --- a python interface for Graphviz

### Usage example

```sh
python3 main.py --input='your file' --output='graph' --alg='coffmat-graham' --params='{\"W\": 2}' && xdg-open 'graph.png'
```
