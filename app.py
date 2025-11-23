import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
st.set_page_config(page_title="BTL VL1")

st.title("Mô hình từ trường của dòng điện tròn")



# Đặt h=mu0/4pi = 1e-7
h=1e-7
I = st.number_input(
            f'Nhập cường độ dòng điện I',
            min_value=0.0, max_value=10.0, step=0.01, 
            value=1.0, key=f"I"
        )
R = st.number_input(
            f'Nhập bán kính dòng điện tròn',
            min_value=0.0, max_value=10.0, step=0.01, 
            value=1.0, key=f"R"
        )
N = int(1000*R)  #  số "đoạn thẳng" chia từ đường tròn

theta = np.linspace(0, 2*np.pi, N, endpoint=False)
#chia thành N góc(tổng N góc là 2pi), endpoint=False để không lặp lại góc 2pi

x = R * np.cos(theta)
y = R * np.sin(theta)
z = np.zeros(N)
#tọa độ x,y,x của N điểm(tạo thành N đoạn thẳng rất nhỏ)
#x,y,z là các ma trận kích thước 1xN

p=np.c_[x, y, z]
#ghép các ma trận x,y,z theo cột tạo thành ma trận p kích thước Nx3


dks = np.array([0, 0, 0])  # tọa độ điểm cần khảo sát vector cảm ứng từ
st.write("Nhập tọa độ điểm khảo sát")
dks[0]=st.number_input(
            f'Hoành độ x',
            min_value=0.0, max_value=10.0, step=0.01, 
            value=0.0, key=f"x"
        )
dks[1]=st.number_input(
            f'Tung độ y',
            min_value=0.0, max_value=10.0, step=0.01, 
            value=0.0, key=f"y"
        )
dks[2]=st.number_input(
            f'Cao độ z',
            min_value=0.0, max_value=10.0, step=0.01, 
            value=0.0, key=f"z"
        )
B = np.zeros(3)   #vector cảm ứng từ cần tìm

for i in range(N): #i chạy từ 0 tới N-1
    dl=p[i]-p[i-1]   
    r=dks-(p[i-1]+p[i])/2 #tổng trung bình
    #r=dks-p[i]   #tổng phải
    # r=dks-p[i-1]  #tỗng trái
    B += np.cross(dl,r)/np.linalg.norm(r)**3
B=B*h*I
if st.button("Tính và hiển thị ảnh"):
    s = f"{np.linalg.norm(B):.2e}"          
    coef, exp = s.split('e')   
    coef = float(coef)         
    exp = int(exp)
    st.success(rf'Độ lớn từ trường tại ({dks[0]},{dks[1]},{dks[2]}) là $\vec{{B}}={coef} \times 10^{{{exp}}}$ T')


    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')###tạo ra bố cục các ô vẽ gồm 1 hàng, 1 cột và chọn ô số 1.
    
    L = np.max(np.abs([R,dks[0], dks[1], dks[2]]))+1
    ax.set_xlim([-L, L])
    ax.set_ylim([-L, L])
    ax.set_zlim([-L, L])
    ax.set_box_aspect([1,1,2])


    ax.plot(x, y, z,c="red")  #đường nối N điểm
    k=-int(N/6)
    ax.quiver(p[k,0],p[k,1],p[k,2], p[k+1,0]-p[k,0],p[k+1,1]-p[k,1],p[k+1,2]-p[k,2],
            length=0.05*R,
            normalize=True,
            arrow_length_ratio=5,
            color='red', label=f"Dòng điện I = {I} A")
    ###3 tham số đầu là điểm đầu và 3 tham số tiếp theo là hướng vector


    ax.scatter(dks[0], dks[1], dks[2],c="green", label=f"Điểm khảo sát ({dks[0]},{dks[1]},{dks[2]})")

    ###vector cảm ứng từ
    ax.quiver(dks[0], dks[1], dks[2], B[0], B[1], B[2], 
            length=L/4, 
            normalize=True,
            arrow_length_ratio=0.5, label=rf'$\vec{{B}}={coef} \times 10^{{{exp}}}$ T')


    plt.legend(loc="upper left")
    st.pyplot(fig)