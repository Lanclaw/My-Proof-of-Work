/*
给定一个 n 个点 m 条边的有向图，图中可能存在重边和自环， 边权可能为负数。

请你求出从 1 号点到 n 号点的最多经过 k 条边的最短距离，如果无法从 1 号点走到 n 号点，输出 impossible。

注意：图中可能 存在负权回路 。

输入格式
第一行包含三个整数 n,m,k。

接下来 m 行，每行包含三个整数 x,y,z，表示存在一条从点 x 到点 y 的有向边，边长为 z。

输出格式
输出一个整数，表示从 1 号点到 n 号点的最多经过 k 条边的最短距离。

如果不存在满足条件的路径，则输出 impossible。

数据范围
1≤n,k≤500,
1≤m≤10000,
任意边长的绝对值不超过 10000。

输入样例：
3 3 1
1 2 1
2 3 1
1 3 3
输出样例：
3
*/

#include<iostream>
#include<cstring>
#include<algorithm>
using namespace std;

const int N = 510, M = 10010;
int n, m, k;
int x, y, z;
int dist[N], backup[N];

struct Edge {
    int x, y, z;
}edges[M];

int bellman_ford() {
    memset(dist, 0x3f, sizeof dist);
    dist[1] = 0;

    for (int i = 0; i < k; i++) {
        memcpy(backup, dist, sizeof dist);
        for (int j = 0; j < m; j++) {
            int a = edges[j].x, b = edges[j].y, w = edges[j].z;
            dist[b] = min(dist[b], backup[a] + w);
        }
    }

    if (dist[n] > 0x3f3f3f3f / 2)   return -1;
    return dist[n];
}

int main() {
    scanf("%d%d%d", &n, &m, &k);

    for (int i = 0; i < m; i++) {
        scanf("%d%d%d", &x, &y, &z);
        edges[i] = { x, y, z };
    }

    int res = bellman_ford();
    if (res == -1)  cout << "impossible";
    else    cout << res;

    return 0;
}