import cv2
import streamlit as st

# Update the sidebar image editing section
if st.session_state.current_image:
    st.subheader("Image Editing")
    
    edit_mode = st.selectbox(
        "Edit Mode",
        ["Basic", "Instruction-Based", "ControlNet", "Image-to-Image"]
    )
    
    if edit_mode == "Basic":
        brightness = st.slider("Brightness", 0.0, 2.0, 1.0)
        contrast = st.slider("Contrast", 0.0, 2.0, 1.0)
        sharpness = st.slider("Sharpness", 0.0, 2.0, 1.0)
        
        if st.button("Apply Basic Edits"):
            edited_image = st.session_state.current_image
            if brightness != 1.0:
                edited_image = image_editor.adjust_brightness(edited_image, brightness)
            if contrast != 1.0:
                edited_image = image_editor.adjust_contrast(edited_image, contrast)
            if sharpness != 1.0:
                edited_image = image_editor.adjust_sharpness(edited_image, sharpness)
            st.session_state.current_image = edited_image
    
    elif edit_mode == "Instruction-Based":
        instruction = st.text_input("Edit Instruction", 
            placeholder="e.g., 'Make it more detailed' or 'Add labels to the diagram'")
        image_guidance = st.slider("Image Guidance Scale", 0.0, 2.0, 1.0)
        
        if st.button("Apply Instruction"):
            with st.spinner("Applying instruction-based edit..."):
                edited_image = image_editor.edit_with_instruction(
                    st.session_state.current_image,
                    instruction,
                    image_guidance_scale=image_guidance
                )
                st.session_state.current_image = edited_image
    
    elif edit_mode == "ControlNet":
        prompt = st.text_input("Edit Prompt", 
            placeholder="e.g., 'High-quality medical illustration with clear labels'")
        control_scale = st.slider("Control Strength", 0.0, 2.0, 1.0)
        
        if st.button("Apply ControlNet"):
            with st.spinner("Applying ControlNet edit..."):
                edited_image = image_editor.edit_with_controlnet(
                    st.session_state.current_image,
                    prompt,
                    controlnet_conditioning_scale=control_scale
                )
                st.session_state.current_image = edited_image
    
    else:  # Image-to-Image
        prompt = st.text_input("Transformation Prompt",
            placeholder="e.g., 'Convert to a detailed anatomical diagram'")
        strength = st.slider("Transformation Strength", 0.0, 1.0, 0.8)
        
        if st.button("Apply Transformation"):
            with st.spinner("Applying image-to-image transformation..."):
                edited_image = image_editor.image_to_image(
                    st.session_state.current_image,
                    prompt,
                    strength=strength
                )
                st.session_state.current_image = edited_image

    # Display current image
    st.image(st.session_state.current_image, use_column_width=True) 