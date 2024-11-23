![SeismoViz Logo](docs/source/_static/logos/seismoviz_banner_light.jpg#gh-light-mode-only)
![SeismoViz Logo](docs/source/_static/logos/seismoviz_banner_dark.jpg#gh-dark-mode-only)


# SeismoViz

SeismoViz is an open-source Python library designed to simplify the analysis, manipulation, and visualization of seismic catalogs. With its intuitive and efficient interface, it reduces complex workflows into just a few lines of code, empowering users to explore seismic data effortlessly.

## ✨ Features

### Catalog refinement
Easily clean and organize your seismic catalog. With SeismoViz, you can remove duplicates, filter data based on your needs, and sort events with just a few simple commands.

```python
import pandas as pd
import seismoviz as sv

# Read the catalog from a file
catalog = sv.read_catalog(path='global_seismic_catalog.csv')

# Remove duplicates
catalog.deduplicate_events()

# Apply filters
fc = catalog.filter(
    mag=('greater', 4),
    depth=('between', [50, 100]),
    time=('lower', '2019-12-28')
)

# Apply sorting
fc.sort(by='time', ascending=True)

# Save the refined catalog
fc.data.to_csv('global_seismic_catalog_filtered.csv', index=False)
```

### Advanced visualization
SeismoViz provides tools to quickly create clear and publication-ready visualizations of seismic data. From plotting earthquake distributions on a map to generating cross-sections, you can easily represent your catalog in a variety of ways.

```python
import seismoviz as sv

# Read the catalog from a file
catalog = sv.read_catalog(path='global_seismic_catalog.csv')

# Create a map showing earthquake locations
catalog.plot_map(
    title='Global seismicity (M > 4.0)',
    color_by='depth',
    cmap='YlOrRd',
    size='mag',
    projection=ccrs.Robinson()
)
```

![Global catalog](docs/source/_images/global_seismicity_catalog.jpg)

```python
import seismoviz as sv

# Read the catalog from a file
catalog = sv.read_catalog(path='local_seismic_catalog.csv')

# Create cross section object
cs = sv.create_cross_section(
    catalog=catalog,        
    center=(13.12, 42.83),  
    num_sections=(0,0),     
    thickness=2,            
    strike=155,             
    map_length=40,          
    depth_range=(0, 10)     
)

# Visualize the cross-section
cs.plot_sections(
    color_by='time',        
    cmap='Blues',           
    size='mag',             
    edgecolor='black'   
)     
```
![Cross section](docs/source/_images/cross_section.jpg)

### Statistical analysis
SeismoViz simplifies seismic data analysis with built-in tools for exploring catalog distributions, plotting magnitude histograms, and calculating the b-value directly from your catalog. These features enable quick and straightforward statistical analysis of your seismic data.



## 📋 Documentation

For a comprehensive guide on how to install, configure, and use this library, please refer to the [SeismoViz official documentation](https://seismoviz.readthedocs.io/en/latest/index.html). It provides detailed instructions on setting up the library, as well as examples and best practices to help you get the most out of your experience.

## 🤝 Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

**NOTE:** Make sure your code contributions follow the Python PEP8 Style Guide standards.

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 📧 Contacts

Author: Gabriele Paoletti  
E-mail: gabriele.paoletti@uniroma1.it

Feel free to reach out if you have any questions or suggestions!