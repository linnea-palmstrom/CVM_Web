# CRESCENT CVM-Web

## Introduction

This is the development code for the [CVM-Tools web interface](https://cvmweb-albfa-xk4p5bggbtl6-1199205512.us-east-2.elb.amazonaws.com/). CVM-Tools is a Python 3 tools package developed to support the Cascadia Community Velocity Model (CVM) working group’s effort in constructing a three-dimensional representation of subsurface material properties for the Cascadia region. These tools facilitate the storage, extraction, and visualization of the CVM through uniform and self-contained data formats.

## Getting Started

This project is dockerized, and you can build and run it on your computer as follows:

### To build:

```bash
docker build -t cvm_web_app .
```

### To run detached:

```bash
docker run -d -p 8000:80 -v $(pwd)/app:/app cvm_web_app
```

### To run and see messages:

```bash
docker run -p 8000:80 -v $(pwd)/app:/app cvm_web_app
```

### Prerequisites

Dependencies are listed in `requirements.txt`.