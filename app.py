import streamlit as st
import numpy as np
st.set_page_config(page_title="BTL VL1")

st.title("Mô hình từ trường của dòng điện tròn")



# Đặt h=mu0/4pi = 1e-7
h=1e-7
I = st.number_input(
            f'Nhập cường độ dòng điện I',
            min_value=0.0, max_value=10, step=0.01, key=f"I"
        )
R = st.number_input(
            f'Nhập bán kính dòng điện tròn',
            min_value=0.0, max_value=10, step=0.01, key=f"R"
        )
N = 1000  #  số "đoạn thẳng" chia từ đường tròn

theta = np.linspace(0, 2*np.pi, N, endpoint=False)
#chia thành N góc(tổng N góc là 2pi), endpoint=False để không lặp lại góc 2pi

x = R * np.cos(theta)
y = R * np.sin(theta)
z = np.zeros(N)
#tọa độ x,y,x của N điểm(tạo thành N đoạn thẳng rất nhỏ)
#x,y,z là các ma trận kích thước 1xN

p=np.c_[x, y, z]
#ghép các ma trận x,y,z theo cột tạo thành ma trận p kích thước Nx3

dks = np.array([0.5, 0, 1])  # tọa độ điểm cần khảo sát vector cảm ứng từ
B = np.zeros(3)   #vector cảm ứng từ cần tìm

for i in range(N): #i chạy từ 0 tới N-1
    dl=p[i]-p[i-1]   
    r=dks-(p[i-1]+p[i])/2 #tổng trung bình
    #r=dks-p[i]   #tổng phải
    # r=dks-p[i-1]  #tỗng trái
    B += I*np.cross(dl,r)/np.linalg.norm(r)**3
B*=h

s = f"{np.linalg.norm(B):.2e}"          
coef, exp = s.split('e')   
coef = float(coef)         
exp = int(exp)
st.success("np.linalg.norm(B)")