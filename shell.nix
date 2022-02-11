{ pkgs ? import <nixpkgs> {} }:
let
  python-with-my-packages = pkgs.python3.withPackages (p: with p; [
    numpy
    sympy
    matplotlib
  ]);
in
python-with-my-packages.env 
