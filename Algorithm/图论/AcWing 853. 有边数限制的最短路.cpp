/*
����һ�� n ���� m ���ߵ�����ͼ��ͼ�п��ܴ����رߺ��Ի��� ��Ȩ����Ϊ������

��������� 1 �ŵ㵽 n �ŵ����ྭ�� k ���ߵ���̾��룬����޷��� 1 �ŵ��ߵ� n �ŵ㣬��� impossible��

ע�⣺ͼ�п��� ���ڸ�Ȩ��· ��

�����ʽ
��һ�а����������� n,m,k��

������ m �У�ÿ�а����������� x,y,z����ʾ����һ���ӵ� x ���� y ������ߣ��߳�Ϊ z��

�����ʽ
���һ����������ʾ�� 1 �ŵ㵽 n �ŵ����ྭ�� k ���ߵ���̾��롣

�������������������·��������� impossible��

���ݷ�Χ
1��n,k��500,
1��m��10000,
����߳��ľ���ֵ������ 10000��

����������
3 3 1
1 2 1
2 3 1
1 3 3
���������
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