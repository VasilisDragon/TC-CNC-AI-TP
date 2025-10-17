#version 330 core

in vec3 v_normal;
in vec3 v_worldPos;

out vec4 fragColor;

uniform vec3 u_lightDir;
uniform vec3 u_color;

void main()
{
    vec3 normal = normalize(v_normal);
    vec3 lightDir = normalize(u_lightDir);
    float diffuse = max(dot(normal, lightDir), 0.1);
    vec3 ambient = vec3(0.15);
    vec3 color = (ambient + diffuse) * u_color;
    fragColor = vec4(color, 1.0);
}

