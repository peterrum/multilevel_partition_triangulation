#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#ifdef DEAL_II_WITH_METIS
#  include <metis.h>
#endif

#include <fstream>

using namespace dealii;

namespace dealii
{
  namespace GridTools
  {
    namespace
    {
      class Graph
      {
      public:
        std::vector<int> xadj;
        std::vector<int> adjncy;
        std::vector<int> weights;
        int              elements;
        std::vector<int> parts;

        void
        print(std::ostream &out)
        {
          std::cout << std::endl;
          std::cout << "Graph:" << std::endl;
          for (auto i : xadj)
            out << i << " ";
          out << std::endl;

          for (auto i : adjncy)
            out << i << " ";
          out << std::endl;

          for (auto i : weights)
            out << i << " ";
          out << std::endl;

          out << elements << std::endl;

          for (auto i : parts)
            out << i << " ";
          out << std::endl << std::endl;
        }
      };



      template <int dim, int spacedim>
      Graph
      create_mesh(const Triangulation<dim, spacedim> &triangulation)
      {
        Graph graph;
        graph.xadj.push_back(0);

        std::map<unsigned int, std::vector<unsigned int>>
                                             coinciding_vertex_groups;
        std::map<unsigned int, unsigned int> vertex_to_coinciding_vertex_group;

        GridTools::collect_coinciding_vertices(
          triangulation,
          coinciding_vertex_groups,
          vertex_to_coinciding_vertex_group);

        for (auto cell : triangulation.active_cell_iterators())
          {
            for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell;
                 ++v)
              {
                auto coinciding_vertex_group =
                  vertex_to_coinciding_vertex_group.find(cell->vertex_index(v));
                if (coinciding_vertex_group !=
                    vertex_to_coinciding_vertex_group.end())
                  graph.adjncy.push_back(coinciding_vertex_group->second);
                else
                  graph.adjncy.push_back(cell->vertex_index(v));
              }
            graph.xadj.push_back(graph.adjncy.size());
          }

        return graph;
      }



      template <int dim>
      Graph
      create_mesh(std::vector<CellData<dim>> &cell_data)
      {
        Graph graph;
        graph.xadj.push_back(0);

        for (auto cell : cell_data)
          {
            for (const auto &v : cell.vertices)
              graph.adjncy.push_back(v);
            graph.xadj.push_back(graph.adjncy.size());
          }

        return graph;
      }



      Graph
      mesh_to_dual_graph_metis(const Graph &      graph_in,
                               const unsigned int n_common_in)
      {
        Graph graph_out;
#ifdef DEAL_II_WITH_METIS
        // extract relevant quantities
        const unsigned int n_elements = graph_in.xadj.size() - 1;
        const unsigned int n_nodes =
          *std::max_element(graph_in.adjncy.begin(), graph_in.adjncy.end()) + 1;

        // convert data type
        idx_t              numflag = 0;
        idx_t              ne      = n_elements;
        idx_t              nn      = n_nodes;
        idx_t              ncommon = n_common_in;
        std::vector<idx_t> eptr    = graph_in.xadj;
        std::vector<idx_t> eind    = graph_in.adjncy;
        ;

        // perform actual conversion from mesh to dual graph
        idx_t *xadj;
        idx_t *adjncy;
        AssertThrow(
          METIS_MeshToDual(
            &ne, &nn, &eptr[0], &eind[0], &ncommon, &numflag, &xadj, &adjncy) ==
            METIS_OK,
          ExcMessage("There has been problem during METIS_MeshToDual."));

        // convert result to the right format
        graph_out.xadj.resize(n_elements + 1);
        for (unsigned int i = 0; i < n_elements + 1; i++)
          graph_out.xadj[i] = xadj[i];

        const unsigned int n_links    = xadj[ne];
        auto &             adjncy_out = graph_out.adjncy;
        adjncy_out.resize(n_links);
        for (unsigned int i = 0; i < n_links; i++)
          adjncy_out[i] = adjncy[i];

        graph_out.parts.resize(n_elements);
        graph_out.elements = n_elements;

        // delete temporal variables
        AssertThrow(METIS_Free(xadj) == METIS_OK,
                    ExcMessage("There has been problem during METIS_Free."));
        AssertThrow(METIS_Free(adjncy) == METIS_OK,
                    ExcMessage("There has been problem during METIS_Free."));
#else
        AssertThrow(false,
                    ExcMessage("deal.II hase not been compiled with Metis."));
        (void)graph_in;
        (void)n_common_in;
#endif

        return graph_out;
      }



      void
      partition_metis(Graph &graph, const unsigned int n_partitions)
      {
#ifdef DEAL_II_WITH_METIS
        idx_t ne   = graph.elements;
        idx_t ncon = 1;
        idx_t edgecut;
        idx_t nparts = n_partitions;

        std::vector<idx_t> xadj   = graph.xadj;
        std::vector<idx_t> adjncy = graph.adjncy;
        std::vector<idx_t> parts(graph.elements);

        int status = METIS_OK;

        if (n_partitions == 1)
          {
            for (unsigned int i = 0; i < parts.size(); i++)
              parts[i] = 0;
          }
        else
          {
            idx_t options[METIS_NOPTIONS];
            METIS_SetDefaultOptions(options);
            // options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_VOL;
            status =
              METIS_PartGraphKway(&ne,
                                  &ncon,
                                  &xadj[0],
                                  &adjncy[0],
                                  NULL,
                                  NULL,
                                  graph.weights.size() == 0 ? NULL :
                                                              &graph.weights[0],
                                  &nparts,
                                  NULL,
                                  NULL,
                                  options,
                                  &edgecut,
                                  &parts[0]);
          }

        AssertThrow(status == METIS_OK,
                    ExcMessage("Partitioning with Metis was not successful."));

        graph.parts = parts;
#else
        AssertThrow(false,
                    ExcMessage("deal.II hase not been compiled with Metis."));
        (void)graph;
        (void)n_partitions;
#endif
      }



      Graph
      mesh_to_dual_graph(const Graph &                    graph_in,
                         const unsigned int               n_common,
                         const SparsityTools::Partitioner partitioner)
      {
        if (partitioner == SparsityTools::Partitioner::metis)
          return mesh_to_dual_graph_metis(graph_in, n_common);

        AssertThrow(false, ExcMessage("Partitioner not known!"));

        return graph_in;
      }



      void
      partition(Graph &                          graph,
                const unsigned int               n_partitions,
                const SparsityTools::Partitioner partitioner)
      {
        if (partitioner == SparsityTools::Partitioner::metis)
          return partition_metis(graph, n_partitions);

        AssertThrow(false, ExcMessage("Partitioner not known!"));
      }



      void
      sort(Graph &graph, const Graph &graph_compressed)
      {
        std::vector<std::pair<unsigned int, unsigned int>> list;
        list.reserve(graph.parts.size());

        for (const auto i : graph.parts)
          list.emplace_back(graph_compressed.parts[i], i);

        std::sort(list.begin(), list.end());
        list.erase(unique(list.begin(), list.end()), list.end());

        for (unsigned int i = 0; i < list.size(); i++)
          list[i].first = i;

        std::sort(list.begin(), list.end(), [](const auto &a, const auto &b) {
          return a.second < b.second;
        });

        for (auto &i : graph.parts)
          i = list[i].first;
      }



      Graph
      compress(const Graph &graph_in)
      {
        const unsigned int n_parts =
          *std::max_element(graph_in.parts.begin(), graph_in.parts.end()) + 1;
        std::vector<std::map<unsigned int, unsigned int>> temp(n_parts);

        Graph graph_out;

        for (unsigned int i = 0; i < graph_in.xadj.size() - 1; i++)
          {
            const unsigned int rank_a = graph_in.parts[i];
            for (int j = graph_in.xadj[i]; j < graph_in.xadj[i + 1]; j++)
              {
                const unsigned int rank_b = graph_in.parts[graph_in.adjncy[j]];

                if (rank_a == rank_b)
                  continue;

                temp[rank_a][rank_b] = 0;
              }
          }

        for (unsigned int i = 0; i < graph_in.xadj.size() - 1; i++)
          {
            const unsigned int rank_a = graph_in.parts[i];
            for (int j = graph_in.xadj[i]; j < graph_in.xadj[i + 1]; j++)
              {
                const unsigned int rank_b = graph_in.parts[graph_in.adjncy[j]];

                if (rank_a == rank_b)
                  continue;

                if (graph_in.weights.size() == 0)
                  temp[rank_a][rank_b]++;
                else
                  temp[rank_a][rank_b] += graph_in.weights[j];
              }
          }

        std::map<unsigned int, unsigned int> counts;

        for (auto &set : temp)
          for (auto &i : set)
            counts[i.second] = 0;

        unsigned int cc = 1; // counts.size();
        for (auto &c : counts)
          c.second = cc++;

        graph_out.xadj.push_back(0);

        for (auto &set : temp)
          {
            for (auto &i : set)
              {
                graph_out.adjncy.push_back(i.first);
                graph_out.weights.push_back(counts[i.second]);
              }
            graph_out.xadj.push_back(graph_out.adjncy.size());
          }

        graph_out.elements = n_parts;

        return graph_out;
      }



      void
      multilevel_partition_triangulation(
        Graph &                          graph,
        const std::vector<unsigned int> &n_partitions,
        const SparsityTools::Partitioner partitioner)
      {
        // pre-partition graph
        partition(graph, n_partitions[0], partitioner);

        if (n_partitions.size() > 1)
          {
            // compress graph according to the previously computed partition
            Graph graph_compressed = compress(graph);

            // partition coarser graph
            multilevel_partition_triangulation(
              graph_compressed,
              std::vector<unsigned int>(n_partitions.begin() + 1,
                                        n_partitions.end()),
              partitioner);

            graph_compressed.print(std::cout);

            // renumerate partitions on graph according to partitions on
            // coarser graph
            sort(graph, graph_compressed);
          }
      }



    } // namespace



    template <int dim, int spacedim>
    void
    multilevel_partition_triangulation(
      const std::vector<unsigned int>  n_partitions,
      Triangulation<dim, spacedim> &   triangulation,
      const SparsityTools::Partitioner partitioner)
    {
      // extract graph from triangulation
      Graph graph = mesh_to_dual_graph(create_mesh(triangulation),
                                       GeometryInfo<dim>::vertices_per_face,
                                       partitioner);

      // perform partitioning on graph
      multilevel_partition_triangulation(graph, n_partitions, partitioner);

      // copy partioning to triangulation
      for (auto cell : triangulation.active_cell_iterators())
        cell->set_manifold_id(graph.parts[cell->index()]);
    }



    template <int dim>
    void
    multilevel_partition_triangulation(
      const std::vector<unsigned int>  n_partitions,
      std::vector<CellData<dim>> &     cell_data,
      const SparsityTools::Partitioner partitioner)
    {
      // extract graph from triangulation
      Graph graph = mesh_to_dual_graph(create_mesh(cell_data),
                                       GeometryInfo<dim>::vertices_per_face,
                                       partitioner);

      // perform partitioning on graph
      multilevel_partition_triangulation(graph, n_partitions, partitioner);

      std::vector<std::tuple<CellData<dim>, unsigned int, unsigned int>> s;

      for (unsigned int i = 0; i < cell_data.size(); ++i)
        s.emplace_back(cell_data[i], graph.parts[i], i);

      std::sort(s.begin(), s.end(), [](const auto &a, const auto &b) {
        if (std::get<1>(a) != std::get<1>(b))
          return std::get<1>(a) < std::get<1>(b);

        return std::get<2>(a) < std::get<2>(b);
      });

      cell_data.clear();

      for (const auto &i : s)
        cell_data.push_back(std::get<0>(i));
    }
  } // namespace GridTools
} // namespace dealii


int
main()
{
  const int                  dim = 3;
  dealii::Triangulation<dim> tria;
  dealii::GridIn<dim>        grid_in(tria);
  std::ifstream              file("triangulation.vtk");
  grid_in.read_vtk(file);

  auto [vertices, cell_data, sub_cell_data] =
    GridTools::get_coarse_mesh_description(tria);

  const std::vector<unsigned int> partitions = {100, 10};

  if (false)
    GridTools::multilevel_partition_triangulation(
      partitions, tria, SparsityTools::Partitioner::metis);
  else
    GridTools::multilevel_partition_triangulation(
      partitions, cell_data, SparsityTools::Partitioner::metis);

  Triangulation<dim> tria_2;
  tria_2.create_triangulation(vertices, cell_data, {});

  unsigned int counter = 0;
  for (const auto &cell : tria_2.active_cell_iterators())
    cell->set_material_id(
      (counter++) / ((tria_2.n_cells() + partitions[0] - 1) / partitions[0]));

  std::ofstream output("test.vtk");
  GridOut       grid_out;
  grid_out.write_vtk(tria_2, output);
}
