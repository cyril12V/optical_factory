{ pkgs }: {
  deps = [
    pkgs.python39
    pkgs.python39Packages.pip
    pkgs.python39Packages.numpy
    pkgs.python39Packages.scikit-learn
  ];
} 