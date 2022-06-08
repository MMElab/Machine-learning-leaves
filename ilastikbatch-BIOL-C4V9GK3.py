# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 09:57:23 2022

@author: vinkjo
This code is 
"""


import subprocess
from pathlib import Path
import os

ilastikpath = "C:\\Program Files\\ilastik-1.3.3post3\\ilastik.exe"
projectpaths = 'C:\\Users\\vinkjo\\'
multifolder = 'C:\\Users\\vinkjo\\OneDrive - Victoria University of Wellington - STAFF\\Desktop\\Machine learning Leaves\\Raw data\\3770'
def batchilastik(folder):
    ## choose whether pixel and/or object projects are run
    pixel = 1
    objectname = 1
    ## choose for which kind of part of the analysis is run
    petri = 1
    plug = 1
    necrosis = 1
    leaf = 1
    
    for filein in folder.glob("*.h5"):
        filename = os.path.basename(filein)
        filename = filename[:-3]
        pixelprobfilename = filename+'_Probabilities.h5'
        if pixel == 1:
            if petri == 1:
                subprocess.run([
                    "C:\\Program Files\\ilastik-1.3.3post3\\ilastik.exe",
                    "--headless",
                    "--output_format=compressed hdf5",
                    "--project=C:\\Users\\vinkjo\\Petridish_pixel.ilp",
                    f"--raw_data={filein}",
                    "--export_source=Probabilities"
                ])
            if plug == 1:
                subprocess.run([
                    "C:\\Program Files\\ilastik-1.3.3post3\\ilastik.exe",
                    "--headless",
                    "--output_format=compressed hdf5",
                    "--project=C:\\Users\\vinkjo\\Plug_pixel.ilp",
                    f"--raw_data={filein}",
                    "--export_source=Probabilities"
                ])
            if necrosis == 1:
                subprocess.run([
                    "C:\\Program Files\\ilastik-1.3.3post3\\ilastik.exe",
                    "--headless",
                    "--output_format=compressed hdf5",
                    "--project=C:\\Users\\vinkjo\\Necrosis_pixel.ilp",
                    f"--raw_data={filein}",
                    "--export_source=Probabilities"
                ])
            if leaf == 1:
                subprocess.run([
                    "C:\\Program Files\\ilastik-1.3.3post3\\ilastik.exe",
                    "--headless",
                    "--output_format=compressed hdf5",
                    "--project=C:\\Users\\vinkjo\\Leaf_pixel.ilp",
                    f"--raw_data={filein}",
                    "--export_source=Probabilities"
                ])
        if objectname == 1:    
            if petri == 1:
                Petrifilename = os.path.dirname(filein) + '\\Pixelprobabilities_petri\\' + pixelprobfilename
                Petrifilename = Path(Petrifilename)
                subprocess.run([
                    "C:\\Program Files\\ilastik-1.3.3post3\\ilastik.exe",
                    "--headless",
                    "--project=C:\\Users\\vinkjo\\Petridish_object_compacted.ilp",
                    "--export_source=Object Identities",
                    f"--raw_data={filein}",
                    f"--prediction_maps={Petrifilename}"
                ])
            if plug ==1:
                Plugfilename = os.path.dirname(filein) + '\\Pixelprobabilities_plug\\' + pixelprobfilename
                Plugfilename = Path(Plugfilename)
                subprocess.run([
                    "C:\\Program Files\\ilastik-1.3.3post3\\ilastik.exe",
                    "--headless",
                    "--project=C:\\Users\\vinkjo\\Plug_object_compacted.ilp",
                    "--export_source=Object Identities",
                    f"--raw_data={filein}",
                    f"--prediction_maps={Plugfilename}"
                ])
            if leaf == 1:
                Leaffilename = os.path.dirname(filein) + '\\Pixelprobabilities_leaf\\' + pixelprobfilename
                Leaffilename = Path(Leaffilename)
                subprocess.run([
                    "C:\\Program Files\\ilastik-1.3.3post3\\ilastik.exe",
                    "--headless",
                    "--project=C:\\Users\\vinkjo\\Leaf_object_compacted.ilp",
                    "--export_source=Object Identities",
                    f"--raw_data={filein}",
                    f"--prediction_maps={Leaffilename}"
                ])
            if necrosis == 1:
                Necrosisfilename = os.path.dirname(filein) + '\\Pixelprobabilities_necrosis\\' + pixelprobfilename
                Necrosisfilename = Path(Necrosisfilename)
                subprocess.run([
                    "C:\\Program Files\\ilastik-1.3.3post3\\ilastik.exe",
                    "--headless",
                    "--project=C:\\Users\\vinkjo\\Necrosis_object_compacted.ilp",
                    "--export_source=Object Identities",
                    f"--raw_data={filein}",
                    f"--prediction_maps={Necrosisfilename}"
                ])

multifolderpath = Path(multifolder)
for folder in multifolderpath.glob("*_h5"):
    batchilastik(folder)
    # folder = Path(folder)
    # filelist = folder.glob("*.h5")
    # files = " ".join(f'"{f}"' for f in filelist)
    
    # subprocess.run(
    # f"""C:\\Program Files\\ilastik-1.3.3post3\\ilastik.exe --headless --output_format=hdf5 
    # --project=C:\\Users\\vinkjo\\Petridish_pixel.ilp --export_source=Probabilities {files}"""
    # )
    
    # subprocess.run(
    # f"""C:\\Program Files\\ilastik-1.3.3post3\\ilastik.exe --headless --output_format=hdf5 
    # --project=C:\\Users\\vinkjo\\Necrosis_pixel.ilp --export_source=Probabilities {files}"""
    # )
    
    # subprocess.run(
    # f"""C:\\Program Files\\ilastik-1.3.3post3\\ilastik.exe --headless --output_format=hdf5 
    # --project=C:\\Users\\vinkjo\\Leaf_pixel.ilp --export_source=Probabilities {files}"""
    # )
    
    # probfolderpath = folder / "Pixelprobabilities_petri"
    # probfilelist = probfolderpath.glob("*.h5")
    # probfiles = " ".join(f'"{f}"' for f in probfilelist)
    
    # subprocess.run(
    # f"C:\\Program Files\\ilastik-1.3.3post3\\ilastik.exe --headless --output_format=hdf5 --project=C:\\Users\\vinkjo\\Petridish_object.ilp --export_source=Probabilities --raw_data {files} --prediction_maps {probfiles}"
    # ,stdout=subprocess.DEVNULL)
    
    
# def batchilastik(folder):

#     folder = Path(folder)
#     for filein in folder.glob("*.h5"):
#         filename = os.path.basename(filein)
#         filename = filename[:-3]
#         pixelprobfilename = filename+'_Probabilities.h5'
        
#         subprocess.run([
#             "C:\\Program Files\\ilastik-1.3.3post3\\ilastik.exe",
#             "--headless",
#             "--output_format=hdf5",
#             "--project=C:\\Users\\vinkjo\\Petridish_pixel.ilp",
#             f"--raw_data={filein}",
#             "--export_source=Probabilities"
#         ])
#         subprocess.run([
#             "C:\\Program Files\\ilastik-1.3.3post3\\ilastik.exe",
#             "--headless",
#             "--output_format=hdf5",
#             "--project=C:\\Users\\vinkjo\\Plug_pixel.ilp",
#             f"--raw_data={filein}",
#             "--export_source=Probabilities"
#         ])
#         subprocess.run([
#             "C:\\Program Files\\ilastik-1.3.3post3\\ilastik.exe",
#             "--headless",
#             "--output_format=hdf5",
#             "--project=C:\\Users\\vinkjo\\Necrosis_pixel.ilp",
#             f"--raw_data={filein}",
#             "--export_source=Probabilities"
#         ])
#         subprocess.run([
#             "C:\\Program Files\\ilastik-1.3.3post3\\ilastik.exe",
#             "--headless",
#             "--output_format=hdf5",
#             "--project=C:\\Users\\vinkjo\\Leaf_pixel.ilp",
#             f"--raw_data={filein}",
#             "--export_source=Probabilities"
#         ])
#         Petrifilename = os.path.dirname(filein) + '\\Pixelprobabilities_petri\\' + pixelprobfilename
#         Petrifilename = Path(Petrifilename)
#         subprocess.run([
#             "C:\\Program Files\\ilastik-1.3.3post3\\ilastik.exe",
#             "--headless",
#             "--project=C:\\Users\\vinkjo\\Petridish_object.ilp",
#             "--export_source=Object Identities",
#             f"--raw_data={filein}",
#             f"--prediction_maps={Petrifilename}"
#         ])
#         Plugfilename = os.path.dirname(filein) + '\\Pixelprobabilities_plug\\' + pixelprobfilename
#         Plugfilename = Path(Plugfilename)
#         subprocess.run([
#             "C:\\Program Files\\ilastik-1.3.3post3\\ilastik.exe",
#             "--headless",
#             "--project=C:\\Users\\vinkjo\\Plug_object.ilp",
#             "--export_source=Object Identities",
#             f"--raw_data={filein}",
#             f"--prediction_maps={Plugfilename}"
#         ])
#         Leaffilename = os.path.dirname(filein) + '\\Pixelprobabilities_leaf\\' + pixelprobfilename
#         Leaffilename = Path(Leaffilename)
#         subprocess.run([
#             "C:\\Program Files\\ilastik-1.3.3post3\\ilastik.exe",
#             "--headless",
#             "--project=C:\\Users\\vinkjo\\Leaf_object.ilp",
#             "--export_source=Object Identities",
#             f"--raw_data={filein}",
#             f"--prediction_maps={Leaffilename}"
#         ])
#         Necrosisfilename = os.path.dirname(filein) + '\\Pixelprobabilities_necrosis\\' + pixelprobfilename
#         Necrosisfilename = Path(Necrosisfilename)
#         subprocess.run([
#             "C:\\Program Files\\ilastik-1.3.3post3\\ilastik.exe",
#             "--headless",
#             "--project=C:\\Users\\vinkjo\\Necrosis_objectv2.ilp",
#             "--export_source=Object Identities",
#             f"--raw_data={filein}",
#             f"--prediction_maps={Necrosisfilename}"
#         ])





# # filelist = glob.glob('C:\\Users\\vinkjo\\Downloads\\OneDrive_2022-03-17\\070322_1_D10_h5\\*.h5')
# # # filelist = ' '.join(filelist)
# filelist = folderpath.glob("*.h5")
# files = " ".join(f'"{f}"' for f in filelist)

# # command = ["C:\\Program Files\\ilastik-1.3.3post3\\ilastik.exe", "--headless", "--output_format", "hdf5", "--project",
# #            "C:\\Users\\vinkjo\\Petridish_pixel.ilp", "--raw_data", "C:\\Users\\vinkjo\\Downloads\\OneDrive_2022-03-17\\070322 1 D10_h5\\IMG_1360.JPG.h5","--export_source","Probabilities"]
# # subprocess.run(command,stdout=subprocess.DEVNULL)
# out_file = os.path.join(folderpath,"{nickname}_probabilities.hf5")
# subprocess.run([
#     "C:\\Program Files\\ilastik-1.3.3post3\\ilastik.exe",
#     "--headless",
#     "--output_format=hdf5",
#     "--project=C:\\Users\\vinkjo\\Petridish_pixel.ilp",
#     #"--raw_data=C:\\Users\\vinkjo\\Downloads\\OneDrive_2022-03-17\\070322 1 D10_h5\\IMG_1360.JPG.h5",
#     #'--raw_data C:\\Users\\vinkjo\\Downloads\\OneDrive_2022-03-17\\070322 1 D10_h5\\IMG_1349.JPG.h5 C:\\Users\\vinkjo\\Downloads\\OneDrive_2022-03-17\\070322 1 D10_h5\\IMG_1351.JPG.h5'
#     #f'--raw_data={files}',
#     #f"--output_filename_format={out_file}",
#     "--export_source=Probabilities",
#     f'{files}'
# ],stdout=subprocess.DEVNULL)