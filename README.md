# fmcfs

fmcfs is a Python library designed for calculating Coulomb stress changes due to fluid injection in geophysical studies.

## Installation

You can install fmcfs using pip:

```bash
conda install maoye::fmcfs
```

## Usage

Here's how you can calculate Coulomb stress using fmcfs:

```python
import fmcfs as fm

data = fm.get_poel("D:\CNresearch\Results\poel_results\mian\d001\inj\pp.t")
print(data.shape)
```

## Features

fmcfs provides:

- Easy-to-use interface for calculating stress changes.
- Support for various fault and source geometries.
- Integration with commonly used geophysical libraries.

## Contributing

Contributions are welcome, especially from those who are also interested in seismology and geophysics.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/NewCalculationModel`)
3. Commit your Changes (`git commit -m 'Add some NewCalculationModel'`)
4. Push to the Branch (`git push origin feature/NewCalculationModel`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Mao Ye - [@maoye](https://twitter.com/maoye) - maoye666@outlook.com

Project Link: [https://github.com/maoye/fmcfs](https://github.com/maoye/coulomb)