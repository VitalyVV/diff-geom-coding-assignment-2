In this environment we use interactive graphics in `Jupyter` based on `THREE.js`. If you already have `python3` and `jupyter` installed, then you should enable `pythreejs` via following commands:

    pip3 install pythreejs
    jupyter nbextension install --py --user pythreejs
    jupyter nbextension enable --py --user pythreejs
    jupyter nbextension enable --py widgetsnbextension

You may find some boilerplate code in `solution.py`, with drawing happening in `draw.py`. Feel free to play with colors etc. In order to test your solution, try in `jupyter`:

    import solution
    mesh = solution.Mesh.fromobj("teddy.obj")
    mesh.draw()

See also `test.ipynb`. The mesh `teddy.obj` is smaller and easier to handle with. Whereas `dragon.obj` will need some optimization.

