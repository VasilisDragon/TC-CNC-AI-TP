#version 330 core

out vec4 fragColor;

uniform vec3 u_color;
uniform float u_alpha;
uniform int u_isDashed;
uniform float u_dashSize;

void main()
{
    if (u_isDashed == 1)
    {
        float stripe = mod(gl_FragCoord.x + gl_FragCoord.y, u_dashSize);
        if (stripe < u_dashSize * 0.5)
        {
            discard;
        }
    }

    fragColor = vec4(u_color, u_alpha);
}

