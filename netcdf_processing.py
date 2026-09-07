
import xarray as xr
import rioxarray
import os
import geopandas as gpd

# Define file paths
input_nc_file = "CROPGRIDSv1.08_cassava.nc"
output_tif_file = "cassava_croparea_sea_2020.tif"

# Define the path to your shapefile containing SEA, PNG, and Bhutan
shapefile_path = r"C:\Users\AFahrezi\OneDrive - CIFOR-ICRAF\FAO\AOI\FAO_RCP51_AOI.shp"

def process_cropgrids(input_file, output_file, shp_path):
    print(f"Loading NetCDF file: {input_file}...")
    
    try:
        # 1. Open the NetCDF dataset using xarray
        # decode_coords="all" ensures spatial dimensions are interpreted correctly
        ds = xr.open_dataset(input_file, decode_coords="all")
        
        # 2. Extract only the 'croparea' (Physical Area) variable
        print("Extracting 'croparea' variable...")
        da = ds['croparea']
        
        # 3. Ensure the CRS is set to WGS 1984 (EPSG:4326)
        # NetCDFs often have this implicitly, but rioxarray needs it explicitly to export to GeoTIFF
        print("Setting CRS to EPSG:4326 (WGS 1984)...")
        da = da.rio.write_crs("epsg:4326")
        
        # Determine the dimension names (sometimes they are 'lon'/'lat', sometimes 'longitude'/'latitude')
        x_dim = 'lon' if 'lon' in da.dims else 'longitude'
        y_dim = 'lat' if 'lat' in da.dims else 'latitude'
        
        # Explicitly set the spatial dimensions for rioxarray
        da = da.rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim)
        
        # 4. Load the shapefile and clip the dataset
        print(f"Loading shapefile: {shp_path}...")
        gdf = gpd.read_file(shp_path)
        
        # Ensure the shapefile CRS matches the dataset CRS (WGS 1984 / EPSG:4326)
        if gdf.crs != "EPSG:4326":
            print("Reprojecting shapefile to EPSG:4326 to match NetCDF...")
            gdf = gdf.to_crs("EPSG:4326")
            
        print("Clipping data to shapefile geometry...")
        # rioxarray 'clip' uses the geometries from the geodataframe. 
        # drop=True ensures the final bounding box shrinks tightly around the shapefile.
        da_clipped = da.rio.clip(gdf.geometry, gdf.crs, drop=True)
        
        # 5. Export to GeoTIFF
        print(f"Exporting to GeoTIFF: {output_file}...")
        # We write null values (NaN) correctly so Earth Engine recognizes NoData
        da_clipped.rio.to_raster(
            output_file, 
            driver="GTiff",
            compress="lzw" # LZW compression reduces file size without losing data
        )
        
        print("Processing complete! Your file is ready for Earth Engine upload.")
        
    except FileNotFoundError:
        print(f"Error: Could not find the file '{input_file}'. Please ensure it is in the same directory as this script.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Clean up memory
        if 'ds' in locals():
            ds.close() # type: ignore

# Execute the function
if __name__ == "__main__":
    # Safety check to ensure both the input files exist before starting
    if os.path.exists(input_nc_file) and os.path.exists(shapefile_path):
        process_cropgrids(input_nc_file, output_tif_file, shapefile_path)
    else:
        print("Error: Could not find the input NetCDF or the Shapefile. Please check the paths.")