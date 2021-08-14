/*
给定一个 n 个点 m 条边的有向图，图中可能存在重边和自环，所有边权均为非负值。

请你求出 1 号点到 n 号点的最短距离，如果无法从 1 号点走到 n 号点，则输出 −1。

输入格式
第一行包含整数 n 和 m。

接下来 m 行每行包含三个整数 x,y,z，表示存在一条从点 x 到点 y 的有向边，边长为 z。

输出格式
输出一个整数，表示 1 号点到 n 号点的最短距离。

如果路径不存在，则输出 −1。

数据范围
1≤n,m≤1.5×105,
图中涉及边长均不小于 0，且不超过 10000。

输入样例：
3 3
1 2 2
2 3 1
1 3 4
输出样例：
3
*/
#include<iostream>
#include<cstring>
#include<queue>
using namespace std;

const int N = 1.5e5 + 10;

typedef pair<int, int> PII;
int h[N], e[N], w[N], ne[N], idx;
bool st[N];
int dist[N];
int n, m;
int x, y, z;

void add(int x, int y, int z) {
    e[idx] = y; w[idx] = z; ne[idx] = h[x]; h[x] = idx++;
}

int dijkstra() {

    memset(dist, 0x3f, sizeof dist);
    dist[1] = 0;
    priority_queue<PII, vector<PII>, greater<PII>> heap;
    heap.push({ 0, 1 });

    while (heap.size()) {
        auto t = heap.top();
        heap.pop();

        int ver = t.second, distance = t.first;
        if (st[ver])    continue;
        st[ver] = true;

        for (int i = h[ver]; i != -1; i = ne[i]) {
            int j = e[i];
            if (dist[j] > distance + w[i]) {
                dist[j] = distance + w[i];
                heap.push({ dist[j], j });
            }
        }

    }

    if (dist[n] == 0x3f3f3f3f)  return -1;
    else return dist[n];

}

int main() {
    scanf("%d%d", &n, &m);
    memset(h, -1, sizeof h);

    while (m--) {
        scanf("%d%d%d", &x, &y, &z);
        add(x, y, z);
    }

    printf("%d", dijkstra());

    return 0;
}
