#version 330 core

in vec3 v_color;

out vec4 fragColor;

uniform float u_alpha;

void main()
{
    fragColor = vec4(v_color, u_alpha);
}
