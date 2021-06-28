"""Dummy to redirect run-as-module to the junn_predict package."""
from junn_predict import __main__

if __name__ == '__main__':
    __main__.main()
