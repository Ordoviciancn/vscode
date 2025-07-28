#include <stdio.h>
#include <stdlib.h>

int main() {
    FILE *file = fopen("Health_Status_and_Functioning.dat", "rb");
    if (file == NULL) {
        perror("Error opening file");
        return 1;
    }
    // 假设文件头是4个字节的整数，表示数据长度
    int data_length;
    fread(&data_length, sizeof(int), 1, file);
    // 根据数据长度读取数据部分（假设是一系列的float数据）
    float *data = (float *)malloc(data_length * sizeof(float));
    fread(data, sizeof(float), data_length, file);
    // 在这里可以对读取的数据进行处理
    fclose(file);
    free(data);
    return 0;
}