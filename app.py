
# Import Modules`
import streamlit as st
from PyNite import FEModel3D
import pandas as pd
import numpy as np

#Analysis function

def beam_analysis(nodes_df,members_df,point_loads_df):


    # Create a new finite element model
    beam = FEModel3D()

    # Define a material
    E = 29000       # Modulus of elasticity (ksi)
    G = 11200       # Shear modulus of elasticity (ksi)
    nu = 0.3        # Poisson's ratio
    rho = 2.836e-4  # Density (kci)
    beam.add_material('Material', E, G, nu, rho)

    
    # Add nodes 
    for i in range(len(nodes_df)):
        beam.add_node(nodes_df.iloc[i,0],nodes_df.iloc[i,1],nodes_df.iloc[i,2],nodes_df.iloc[i,3])

    # Add  beams:
    for i in range(len(members_df)):
        beam.add_member(members_df.iloc[i,0], members_df.iloc[i,1], members_df.iloc[i,2], members_df.iloc[i,3], members_df.iloc[i,4], members_df.iloc[i,5], members_df.iloc[i,6], members_df.iloc[i,7])

    # Provide simple supports
    for i in range(len(nodes_df)):
        beam.def_support(nodes_df.iloc[i,0], nodes_df.iloc[i,4], nodes_df.iloc[i,5], nodes_df.iloc[i,6], nodes_df.iloc[i,7], nodes_df.iloc[i,8], nodes_df.iloc[i,9])  

    # Add point loads to member
    for i in range(len(point_loads_df)):
        beam.add_member_pt_load(point_loads_df.iloc[i,0],point_loads_df.iloc[i,1],point_loads_df.iloc[i,2],point_loads_df.iloc[i,3]) # 5 kips Dead load

    #add_member_dist_load(member_name, Direction, w1, w2, x1=None, x2=None, case='Case 1')
    for i in range(len(udl_loads_df)):
        beam.add_member_dist_load(udl_loads_df.iloc[i,0],udl_loads_df.iloc[i,1],udl_loads_df.iloc[i,2],udl_loads_df.iloc[i,3],udl_loads_df.iloc[i,4],udl_loads_df.iloc[i,5]) # 5 kips Dead load4

    # Analyze the beam and perform a statics check
    beam.analyze(check_statics=True)
           
    return beam

# Display function for plots
def display_plots(beam,members_df):
    for i in range(len(members_df)):
        st.pyplot(beam.Members[members_df.iloc[i,0]].plot_shear('Fy'))
        st.pyplot(beam.Members[members_df.iloc[i,0]].plot_moment('Mz'))
        st.pyplot(beam.Members[members_df.iloc[i,0]].plot_deflection('dy'))
        
def results_table(beam,members_df,nodes_df):
    results_df=pd.DataFrame()
    Mz=[]
    Fy=[]
    dy=[]
    y=[]
    for i in range(len(members_df)):
        #for j in range((nodes_df.iloc[i+1,1]-nodes_df.iloc[i,1])*10):
        x=np.linspace(0,(nodes_df.iloc[i+1,1]-nodes_df.iloc[i,1]),100)
        for xs in x:
            Mz.append(beam.Members[members_df.iloc[i,0]].moment('Mz',xs))
            Fy.append(beam.Members[members_df.iloc[i,0]].shear('Fy',xs))
            dy.append(beam.Members[members_df.iloc[i,0]].deflection('dy',xs))
            y.append(xs+nodes_df.iloc[i,1])

    results_df['y']=y
    results_df['Mz']=Mz
    results_df['Fy']=Fy
    results_df['dy']=dy
    results_df['beam']=0
    
    

    return results_df
    
#Table Data for nodes
node_1 = {'name': 'Node_A', 'x': 0,'y':0,'z':0,'R_Fx':True,'R_Fy':True,'R_Fz':True,'R_Mx':True,'R_My':False,'R_Mz':False}
node_2 = {'name': 'Node_B', 'x': 5,'y':0,'z':0,'R_Fx':True,'R_Fy':True,'R_Fz':True,'R_Mx':False,'R_My':False,'R_Mz':False}
node_3 = {'name': 'Node_C', 'x': 10,'y':0,'z':0,'R_Fx':True,'R_Fy':True,'R_Fz':True,'R_Mx':False,'R_My':False,'R_Mz':False}
node_4 = {'name': 'Node_D', 'x': 15,'y':0,'z':0,'R_Fx':True,'R_Fy':True,'R_Fz':True,'R_Mx':False,'R_My':False,'R_Mz':False}
nodes_df = pd.DataFrame(columns=['name','x','y','z','R_Fx','R_Fy','R_Fz','R_Mx','R_My','R_Mz'])
nodes_df.loc[len(nodes_df)] = node_1
nodes_df.loc[len(nodes_df)] = node_2
nodes_df.loc[len(nodes_df)] = node_3
nodes_df.loc[len(nodes_df)] = node_4



#Beam Table
member_1 = {'Name': 'M1', 'Node_LHS':'Node_A','Node_RHS':'Node_B','Material':'Material','Iy':0.01,'Iz':0.01,'J':0.1,'A':1,}
member_2 = {'Name': 'M2', 'Node_LHS':'Node_B','Node_RHS':'Node_C','Material':'Material','Iy':0.01,'Iz':0.01,'J':0.1,'A':1,}
member_3 = {'Name': 'M3', 'Node_LHS':'Node_C','Node_RHS':'Node_D','Material':'Material','Iy':0.01,'Iz':0.01,'J':0.1,'A':1,}
members_df = pd.DataFrame(columns=['Name','Node_LHS','Node_RHS','Material','Iy','Iz','J','A'])
members_df.loc[len(members_df)] = member_1
members_df.loc[len(members_df)] = member_2
members_df.loc[len(members_df)] = member_3


#Member loads table
#Point Load Table
load_1 = {'Member': 'M1', 'Load':'Fy','Value':-5,'Location':2.5}
load_2 = {'Member': 'M2', 'Load':'Fy','Value':-5,'Location':2.5}
load_3 = {'Member': 'M3', 'Load':'Fy','Value':-5,'Location':2.5}
point_loads_df = pd.DataFrame(columns=['Member','Load','Value','Location',])
point_loads_df.loc[len(point_loads_df)] = load_1
point_loads_df.loc[len(point_loads_df)] = load_2
point_loads_df.loc[len(point_loads_df)] = load_3


#UDL Load Table
load_1 = {'Member': 'M1', 'Load':'Fy','w1':-5,'w2':-5,'x1':0,'x2':5}
load_2 = {'Member': 'M2', 'Load':'Fy','w1':-5,'w2':-5,'x1':0,'x2':5}
load_3 = {'Member': 'M3', 'Load':'Fy','w1':-5,'w2':-5,'x1':0,'x2':5}
udl_loads_df = pd.DataFrame(columns=['Member','Load','w1','w2','x1','x2'])
udl_loads_df.loc[len(udl_loads_df)] = load_1
udl_loads_df.loc[len(udl_loads_df)] = load_2
udl_loads_df.loc[len(udl_loads_df)] = load_3


#Page config

st.set_page_config(page_title="Civils Optimisation",page_icon=":computer",layout="wide")
with st.sidebar:
    st.header('Infomation')
    st.write(st.__version__)
    st.subheader('About')
    st.write('This calculator is run on Pynite FE3Dmodel(). Ensure Nodes match members before calculating results.')
    st.subheader('Notation')
    st.write('When adding element to tables ensure the index values are correct and run from 0.')
    #st.subheader('Sign convention')
    #st.image('images/sign_convention.png')
    st.subheader('Next Steps')
    st.write("Improve plots to show nodes/supports.")
    st.write("Add deflection plots.")
    st.subheader('Author')
    st.write("Tom Cartigny")
        
st.title("Beam Force Diagram Calculator")
st.header("Inputs")
st.set_option('deprecation.showPyplotGlobalUse', False)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Nodes / Restraints")
    nodes_df=st.data_editor(nodes_df, hide_index=True, num_rows="dynamic",column_config={'y':None,'z':None})
    #hide_index=True, column_config={"B": None}
    st.subheader("Members")
    members_df=st.data_editor(members_df,hide_index=True, num_rows="dynamic", column_config={'J':None})

with col2:
    st.subheader("Loads")
    st.subheader("Point Loads")
    point_loads_df=st.data_editor(point_loads_df, hide_index=True, num_rows="dynamic")
    st.subheader("UDL Loads")
    udl_loads_df=st.data_editor(udl_loads_df, hide_index=True, num_rows="dynamic")

if st.button("Calculate"):
    # Perform beam analysis
    beam = beam_analysis(nodes_df, members_df, point_loads_df)
    # Display results
    #display_plots(beam,members_df)
    df=results_table(beam,members_df,nodes_df)
    st.header('Results')
    col1, col2, col3 = st.columns([1,2,2])
    with col1:
        st.subheader('Table')
        st.dataframe(df)
    with col2:
        st.subheader('Moment diagram')
        st.line_chart(df, x='y', y=['beam','Mz'])
    with col3:
        st.subheader('Shear Force Diagram')
        st.line_chart(df, x='y', y=['beam','Fy'])
    st.subheader('Deflection Plot')
    st.line_chart(df, x='y', y=['beam','dy'])