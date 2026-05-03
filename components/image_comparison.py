# Original work Copyright (c) 2021 fatih 
# Modified from the original source by Qingzhao Yan on 2026/05/03.
# 
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
#
# This file has been modified from the original version found in:
# https://github.com/fcakyon/streamlit-image-comparison

import streamlit as st

from streamlit.delta_generator import DeltaGenerator
from typing import Union, Literal

def image_comparison(
	img1: Union[str, bytes],
	img2: Union[str, bytes],
	label1: str = "1",
	label2: str = "2",
	width: Union[int, Literal["stretch", "content"]] = "stretch",
	height: Union[int, Literal["stretch", "content"]] = "content",
	show_labels: bool = True,
	starting_position: int = 50,
	make_responsive: bool = True,
) -> DeltaGenerator:
	"""
	Create a comparison slider for two images.
	
	Parameters
	----------
	img1: base64 encoded img str
		Data for the first image.
	img2: base64 encoded img str
		Data for the second image.
	label1: str, optional
		Label for the first image. Default is "1".
	label2: str, optional
		Label for the second image. Default is "2".
	width: "stretch", "content" or int, optional
		Width of the component. Default is "stretch".
	height: "stretch", "content" or int, optional
		Height of the component. Default is "content".
	show_labels: bool, optional
		Whether to show labels on the images. Default is True.
	starting_position: int, optional
		Starting position of the slider as a percentage (0-100). Default is 50.
	make_responsive: bool, optional
		Whether to enable responsive mode. Default is True.

	Returns
	-------
	st.iframe
		Returns a static component with a timeline
	"""
	
	css_width = f"{width}px" if isinstance(width, int) else "100%"
	css_height = f"{height}px" if isinstance(height, int) else "auto"

	# Load CSS and JS
	cdn_path = "https://cdn.knightlab.com/libs/juxtapose/latest"
	css_block = f'<link rel="stylesheet" href="{cdn_path}/css/juxtapose.css">'
	js_block = f'<script src="{cdn_path}/js/juxtapose.min.js"></script>'

	# write html block
	htmlcode = f"""
		<style>body {{ margin: unset; }}</style>
		{css_block}
		{js_block}
		<div id="foo" style="width: {css_width}; height: {css_height};"></div>
		<script>
		slider = new juxtapose.JXSlider('#foo',
			[
				{{
					src: '{img1}',
					label: '{label1}',
				}},
				{{
					src: '{img2}',
					label: '{label2}',
				}}
			],
			{{
				animate: true,
				showLabels: {'true' if show_labels else 'false'},
				showCredits: true,
				startingPosition: "{starting_position}%",
				makeResponsive: {'true' if make_responsive else 'false'},
			}});
		</script>
		"""
	static_component = st.iframe(htmlcode, height=height, width=width)

	return static_component