import streamlit as st
from PIL import Image
import os

def about_page():
    # Page title and description
    st.markdown(
        """
        <div style='text-align: center; padding: 2em; background-color: #3c007a; color: white; border-radius: 10px;'>
            <h2 style='font-size: 2.5em;'>Our Team</h2>
            <p style='font-size: 1.2em; line-height: 1.5;'>
                This project was carried out by highly skilled students from diverse backgrounds, including finance, engineering, and more, all committed to applying their knowledge to a comprehensive and effective analysis of social network data.
            </p>
        </div>
        """, unsafe_allow_html=True
    )

    # Define team members and their images (relative paths to the 'images' folder)
    team_members = [
        {"name": "Antonio Torreskelly Espinosa", "image_path": "images/antonio.png"},
        {"name": "Zaida Movellan Magaldi", "image_path": "images/zaida.png"},
        {"name": "Sofia Karime Álvarez Falcón", "image_path": "images/sofia.png"},
        {"name": "Julio Cesar José Perez Saavedra", "image_path": "images/julio.png"},
        {"name": "Lucas Alberto May Meneses", "image_path": "images/lucas.png"},
        {"name": "Thomas Origel Morales", "image_path": "images/tom.png"},
    ]

    st.markdown("<br>", unsafe_allow_html=True)  # Spacer for better visual separation

    # Display team members in a grid
    cols_per_row = 3  # Number of team members per row
    rows = len(team_members) // cols_per_row + int(len(team_members) % cols_per_row > 0)

    for row in range(rows):
        cols = st.columns(cols_per_row)
        for i in range(cols_per_row):
            member_index = row * cols_per_row + i
            if member_index < len(team_members):
                member = team_members[member_index]
                with cols[i]:
                    try:
                        # Open and display the image
                        image = Image.open(member["image_path"])
                        st.image(image, caption=member["name"], use_column_width=True)
                    except FileNotFoundError:
                        st.error(f"Image not found for {member['name']}.")
                    except Exception as e:
                        st.error(f"Error loading image for {member['name']}: {e}")

    st.markdown(
        """
        <div style='text-align: center; margin-top: 2em;'>
            <p style='font-size: 1.2em; line-height: 1.5; color: white;'>
                We are a team of enthusiastic professionals working together to create meaningful insights from social data!
            </p>
        </div>
        """, unsafe_allow_html=True
    )


# Call the function to render the page
if __name__ == "__main__":
    about_page()
