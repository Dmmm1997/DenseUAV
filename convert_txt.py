
source_txt = "/home/dmmm/Dataset/DenseUAV/DenseUAV/Dense_GPS_test.txt"
output_txt = "/home/dmmm/Dataset/DenseUAV/DenseUAV/Dense_GPS_test_tmp.txt"

write_file = open(output_txt,"w")

with open(source_txt,"r") as F:
    lines = F.readlines()
    for line in lines:
        info_list = line.split()
        info_list[0] = "/".join(info_list[0].split("/")[-4:])
        str_tmp = " ".join(info_list)
        str_tmp+="\n"
        write_file.write(str_tmp)

write_file.close()
