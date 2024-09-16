from mmcls.datasets import SubwayDataset
csv_file = 'C:\\Users\\55479\\Downloads\\documents-export-2023-9-5\\data\\subway\\dms1118c_clean.csv'   # recommend use abs path
out_dir = 'C:\\Users\\55479\\Downloads\\documents-export-2023-9-5\\data\\subway\\out.csv'
SubwayDataset.split_data(csv_file, out_dir, train_ratio=0.1, shuffle=True)